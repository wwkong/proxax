"""Proximal Augmented Lagrangian with Semismooth Newton (`QPALM`) solver."""

from __future__ import annotations

import dataclasses

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from jax.experimental import sparse as jsparse

from proxax import _typing
from proxax import problem as problem_mod
from proxax import tree
from proxax.constraints import linear_op
from proxax.prox import base as prox_base
from proxax.solvers import base
from proxax.solvers.constrained import _sparse_ldl_ffi

_BracketState = tuple[jax.Array, jax.Array, jax.Array]
_SecantState = tuple[
    jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
]
_WoodburyStep = tuple[jax.Array, jax.Array, jax.Array]
_WoodburyInnerState = tuple[
    jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
]
_QuadInnerState = tuple[
    jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
]
_InnerState = tuple[jax.Array, jax.Array, jax.Array]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class QPALMState(base.ConstrainedTrackerMixin):
  """Outer Augmented Lagrangian iteration state of `QPALM`."""

  primal: _typing.PyTree
  multiplier: _typing.PyTree
  prox_multiplier: _typing.PyTree
  sigma_c: _typing.PyTree
  sigma_x: _typing.PyTree
  prev_e_c: _typing.PyTree
  prev_e_x: _typing.PyTree
  gamma: jax.Array
  inner_tol: jax.Array
  tracker: base.ConstrainedTracker


def _to_flat_array(x: _typing.PyTree) -> tuple[jax.Array, callable]:
  """Flattens a PyTree into a 1D array and returns `(flat, unflatten)`."""
  if isinstance(x, jax.Array) and x.ndim == 1:
    return x, lambda v: v
  return jax.flatten_util.ravel_pytree(x)


def _project_bound_c(
    bound_c: linear_op.BoundConstraint,
    z: _typing.PyTree,
) -> _typing.PyTree:
  """Projects `z` onto `bound_c`'s target set (single or product constraint)."""
  if hasattr(bound_c, "target_set"):
    return bound_c.target_set.project(z)
  return jax.tree.map(
      lambda c, zi: c.target_set.project(zi),
      bound_c.constraints,
      z,
      is_leaf=linear_op._is_constraint,
  )


def _diag_at_w_a(
    bound_c: linear_op.BoundConstraint,
    w_c: _typing.PyTree,
    n: int,
    dtype: jnp.dtype,
) -> jax.Array:
  """Computes `diag(A^T diag(w_c) A)` when `bound_c` is matrix-backed."""
  if hasattr(bound_c, "operator") and isinstance(
      bound_c.operator, linear_op.MatrixLinearOperator
  ):
    mat = bound_c.operator.matrix
    w_arr, _ = _to_flat_array(w_c)
    if isinstance(mat, jax.Array):
      return jnp.sum((w_arr[:, None] * mat) * mat, axis=0)
    if isinstance(mat, jsparse.BCOO):
      rows = mat.indices[:, 0]
      cols = mat.indices[:, 1]
      contrib = w_arr[rows] * (mat.data * mat.data)
      return jnp.zeros((n,), dtype=dtype).at[cols].add(contrib)
  return jnp.zeros((n,), dtype=dtype)


def _exact_1d_linesearch_cached(
    psi_prime_fn,
    psi0: jax.Array,
    *,
    dtype: jnp.dtype,
    max_iters: int = 25,
) -> jax.Array:
  """Exact 1D line search along `d` using a precomputed `psi'(tau)` closure."""
  tau_one = jnp.asarray(1.0, dtype=dtype)
  psi1 = psi_prime_fn(tau_one)
  tol = jnp.asarray(1e-10, dtype=dtype) * jnp.maximum(
      jnp.abs(psi0), jnp.asarray(1.0, dtype=dtype)
  )

  def _do_search(_: None) -> jax.Array:
    def _bracket_cond(st: _BracketState) -> jax.Array:
      _, psi_hi, k = st
      return jnp.logical_and(psi_hi < 0.0, k < 20)

    def _bracket_body(st: _BracketState) -> _BracketState:
      t_hi, _, k = st
      t_next = 2.0 * t_hi
      return (t_next, psi_prime_fn(t_next), k + 1)

    t_hi0, psi_hi0, _ = jax.lax.while_loop(
        _bracket_cond,
        _bracket_body,
        (tau_one, psi1, jnp.asarray(0, dtype=jnp.int32)),
    )
    t_lo0 = jnp.where(psi1 >= 0.0, jnp.asarray(0.0, dtype=dtype), tau_one)
    psi_lo0 = jnp.where(psi1 >= 0.0, psi0, psi1)

    def _sec_cond(st: _SecantState) -> jax.Array:
      t_lo, psi_lo, t_hi, psi_hi, t_mid, k = st
      min_step = jnp.asarray(1e-12, dtype=dtype) * jnp.maximum(t_hi, 1.0)
      not_done = jnp.logical_and(
          jnp.abs(psi_hi - psi_lo) > 0.0,
          (t_hi - t_lo) > min_step,
      )
      psi_mid = jnp.where(k == 0, psi_hi, psi_prime_fn(t_mid))
      return jnp.logical_and(
          k < max_iters,
          jnp.logical_and(not_done, jnp.abs(psi_mid) > tol),
      )

    def _sec_body(st: _SecantState) -> _SecantState:
      t_lo, psi_lo, t_hi, psi_hi, _, k = st
      denom = psi_hi - psi_lo
      t_sec = t_lo - psi_lo * (t_hi - t_lo) / jnp.where(
          jnp.abs(denom) > 1e-30, denom, jnp.asarray(1.0, dtype=dtype)
      )
      margin = 0.05 * (t_hi - t_lo)
      in_bounds = jnp.logical_and(t_sec > t_lo + margin, t_sec < t_hi - margin)
      t_mid = jnp.where(in_bounds, t_sec, 0.5 * (t_lo + t_hi))
      psi_mid = psi_prime_fn(t_mid)
      go_right = psi_mid < 0.0
      t_lo_n = jnp.where(go_right, t_mid, t_lo)
      psi_lo_n = jnp.where(go_right, psi_mid, psi_lo)
      t_hi_n = jnp.where(go_right, t_hi, t_mid)
      psi_hi_n = jnp.where(go_right, psi_hi, psi_mid)
      return (t_lo_n, psi_lo_n, t_hi_n, psi_hi_n, t_mid, k + 1)

    t_lo_f, _, t_hi_f, _, t_mid_f, k_f = jax.lax.while_loop(
        _sec_cond,
        _sec_body,
        (
            t_lo0,
            psi_lo0,
            t_hi0,
            psi_hi0,
            tau_one,
            jnp.asarray(0, dtype=jnp.int32),
        ),
    )
    return jnp.where(k_f > 0, t_mid_f, 0.5 * (t_lo_f + t_hi_f))

  return jax.lax.cond(
      jnp.logical_or(psi0 >= 0.0, jnp.abs(psi1) <= tol),
      lambda _: jnp.where(psi0 >= 0.0, jnp.asarray(0.0, dtype=dtype), tau_one),
      _do_search,
      None,
  )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class QPALM:
  """Proximal Augmented Lagrangian with Semismooth Newton (`QPALM`).

  Solves constrained problems `min_x f(x) + h(x) s.t. A(x) in C` (where `h`
  is `Zero` or an indicator such as `Box`) using exact semismooth Newton steps
  with exact 1D line search on each proximal Augmented Lagrangian subproblem
  and per-constraint penalty updates.
  """

  opt_tol: float = 1e-6
  feas_tol: float = 1e-6
  rel_opt_tol: float = 0.0
  rel_feas_tol: float = 0.0
  norm_kind: str = "linf"
  sigma_init: float = 20.0
  sigma_factor: float = 50.0
  sigma_max: float = 1e8
  theta_feas: float = 0.25
  gamma_init: float = 1e7
  gamma_factor: float = 10.0
  gamma_max: float = 1e8
  inner_tol_init: float = 1e-1
  inner_tol_factor: float = 0.1
  max_alm_iters: int = 1000
  max_inner_iters: int = 100
  linear_solver: str = "auto"
  woodbury_max_rank: int = 16
  assume_quadratic: bool = True
  cg_tol: float = 1e-10
  cg_maxiter: int = 500
  equilibrate: bool = False
  equilibrate_iters: int = 10

  def solve_state(
      self,
      problem: problem_mod.CompositeProblem,
      x0: _typing.PyTree,
      *,
      dual0: _typing.PyTree | None = None,
      jit: bool = True,
      callback: _typing.CallbackFn[QPALMState] | None = None,
      time_limit: float | None = None,
  ) -> QPALMState:
    """Runs `QPALM` and returns the full `QPALMState`."""
    if not problem.is_constrained:
      raise ValueError("QPALM requires a constrained `CompositeProblem`.")
    if time_limit is not None and time_limit < 0:
      raise ValueError("`time_limit` must be non-negative.")

    x_init = tree.as_float_tree(x0)
    dtype = tree.infer_dtype(x_init)
    x_flat0, unflatten = _to_flat_array(x_init)
    n = int(x_flat0.shape[0])

    bound_c = problem.bind_constraints(x_init)
    assert bound_c is not None
    has_prox = not isinstance(problem.proximal, prox_base.Zero)

    def _f_val_and_grad(x_arr: jax.Array) -> tuple[jax.Array, jax.Array]:
      x_pt = unflatten(x_arr)
      val, g_pt = problem.smooth.value_and_grad(x_pt)
      g_arr, _ = _to_flat_array(g_pt)
      return jnp.asarray(val, dtype=dtype), g_arr

    def _f_grad(x_arr: jax.Array) -> jax.Array:
      _, g_arr = _f_val_and_grad(x_arr)
      return g_arr

    use_woodbury = self.linear_solver == "woodbury"
    use_sparse_cb = self.linear_solver == "sparse_callback"
    use_qdldl_ffi = self.linear_solver == "qdldl_ffi"
    use_chol = (
        self.linear_solver == "cholesky"
        or use_woodbury
        or (self.linear_solver == "auto" and n <= 1024)
    )
    a_op_mat = (
        bound_c.operator.matrix
        if (
            (use_chol or use_sparse_cb or use_qdldl_ffi)
            and hasattr(bound_c, "operator")
            and isinstance(bound_c.operator, linear_op.MatrixLinearOperator)
        )
        else None
    )
    a_is_bcoo = use_qdldl_ffi and isinstance(a_op_mat, jsparse.BCOO)
    a_dense = a_op_mat if isinstance(a_op_mat, jax.Array) else None
    a_mat = a_op_mat if (a_dense is not None or a_is_bcoo) else None
    if (
        (use_chol or use_sparse_cb or use_qdldl_ffi)
        and self.assume_quadratic
        and a_mat is not None
    ):
      h_f_const = jax.hessian(lambda v: _f_val_and_grad(v)[0])(x_flat0)
      h_f_const = 0.5 * (h_f_const + h_f_const.T)
      q_lin = _f_grad(x_flat0) - h_f_const @ x_flat0
    else:
      h_f_const = None
      q_lin = None

    if use_sparse_cb and h_f_const is not None and a_dense is not None:
      def _sparse_solve_cb(h_f_np, a_np, w_c_np, diag_add_np, rhs_np):
        h_f_csc = sp.csc_matrix(np.asarray(h_f_np, dtype=np.float64))
        a_csc = sp.csc_matrix(np.asarray(a_np, dtype=np.float64))
        w_c_arr = np.asarray(w_c_np, dtype=np.float64)
        diag_arr = np.asarray(diag_add_np, dtype=np.float64)
        rhs_arr = np.asarray(rhs_np, dtype=np.float64)
        h_csc = (
            h_f_csc
            + a_csc.T @ (sp.diags(w_c_arr) @ a_csc)
            + sp.diags(diag_arr)
        )
        return spla.spsolve(h_csc, rhs_arr).astype(np.asarray(rhs_np).dtype)
    else:
      _sparse_solve_cb = None

    def _proj_x(z_arr: jax.Array) -> jax.Array:
      if not has_prox:
        return z_arr
      p_pt = problem.proximal.prox(
          unflatten(z_arr), jnp.asarray(1.0, dtype=dtype)
      )
      p_arr, _ = _to_flat_array(p_pt)
      return p_arr

    is_equilibrated = bool(
        self.equilibrate and h_f_const is not None and a_mat is not None
    )
    if is_equilibrated:
      m_c = int(a_mat.shape[0])
      d_x = jnp.ones((n,), dtype=dtype)
      d_y = jnp.ones((m_c,), dtype=dtype)
      cost_scale = jnp.ones((), dtype=dtype)
      h_abs = jnp.abs(h_f_const)
      if a_is_bcoo:
        a_rows = a_mat.indices[:, 0]
        a_cols = a_mat.indices[:, 1]
        a_abs = jnp.abs(a_mat.data)
      else:
        a_abs_dense = jnp.abs(a_mat)

      for _ in range(int(self.equilibrate_iters)):
        h_col_norm = jnp.max(
            (cost_scale * d_x[:, None]) * h_abs * d_x[None, :], axis=0
        )
        if a_is_bcoo:
          scaled_a = d_y[a_rows] * a_abs * d_x[a_cols]
          a_row_norm = jnp.zeros((m_c,), dtype=dtype).at[a_rows].max(scaled_a)
          a_col_norm = jnp.zeros((n,), dtype=dtype).at[a_cols].max(scaled_a)
        else:
          scaled_a = d_y[:, None] * a_abs_dense * d_x[None, :]
          a_row_norm = jnp.max(scaled_a, axis=1)
          a_col_norm = jnp.max(scaled_a, axis=0)
        col_norm = jnp.maximum(h_col_norm, a_col_norm)
        s_x = jnp.where(
            col_norm > 0.0,
            1.0 / jnp.sqrt(jnp.maximum(col_norm, 1e-8)),
            1.0,
        )
        s_y = jnp.where(
            a_row_norm > 0.0,
            1.0 / jnp.sqrt(jnp.maximum(a_row_norm, 1e-8)),
            1.0,
        )
        d_x = d_x * s_x
        d_y = d_y * s_y
        h_inf = jnp.max((cost_scale * d_x[:, None]) * h_abs * d_x[None, :])
        q_inf = jnp.max(jnp.abs(cost_scale * d_x * q_lin))
        c_norm = jnp.maximum(h_inf, q_inf)
        gamma_c = jnp.where(c_norm > 0.0, 1.0 / jnp.maximum(c_norm, 1e-8), 1.0)
        cost_scale = cost_scale * gamma_c

      h_f_const = (cost_scale * d_x[:, None]) * h_f_const * d_x[None, :]
      h_f_const = 0.5 * (h_f_const + h_f_const.T)
      q_lin = cost_scale * d_x * q_lin
      if a_is_bcoo:
        a_mat = jsparse.BCOO(
            (a_mat.data * d_y[a_rows] * d_x[a_cols], a_mat.indices),
            shape=a_mat.shape,
        )
      else:
        a_mat = d_y[:, None] * a_mat * d_x[None, :]
        a_dense = a_mat

      def _proj_c_work(z: _typing.PyTree) -> _typing.PyTree:
        return d_y * _project_bound_c(bound_c, z / d_y)

      def _proj_x_work(z_arr: jax.Array) -> jax.Array:
        return _proj_x(d_x * z_arr) / d_x

      def _f_grad_work(x_arr: jax.Array) -> jax.Array:
        return cost_scale * d_x * _f_grad(d_x * x_arr)
    else:
      d_x = jnp.ones((n,), dtype=dtype)
      d_y = jnp.ones((), dtype=dtype)
      cost_scale = jnp.ones((), dtype=dtype)
      def _proj_c_work(z: _typing.PyTree) -> _typing.PyTree:
        return _project_bound_c(bound_c, z)
      _proj_x_work = _proj_x
      _f_grad_work = _f_grad

    mult0 = (
        bound_c.zero_dual(x_init)
        if dual0 is None
        else tree.as_float_tree(dual0)
    )
    prox_mult0 = jnp.zeros_like(x_flat0)
    sigma_c0 = jax.tree.map(lambda a: jnp.full_like(a, self.sigma_init), mult0)
    sigma_x0 = jnp.full_like(x_flat0, self.sigma_init)

    g0_f = _f_grad(x_flat0)
    ax0_orig = bound_c.apply(x_init)
    proj_ax0 = _project_bound_c(bound_c, ax0_orig)
    e_c0 = tree.sub(ax0_orig, proj_ax0)
    e_x0 = x_flat0 - _proj_x(x_flat0)
    v0 = tree.add(unflatten(g0_f), bound_c.adjoint(mult0))
    opt_res0 = tree.norm(v0, kind=self.norm_kind)
    feas_res0 = jnp.maximum(
        tree.norm(e_c0, kind=self.norm_kind),
        tree.norm(e_x0, kind=self.norm_kind),
    )
    obj0, _ = _f_val_and_grad(x_flat0)

    tracker0 = base.ConstrainedTracker.from_init(
        x_init,
        mult0,
        v0,
        opt_res0,
        feas_res0,
        obj0,
        opt_tol=self.opt_tol,
        feas_tol=self.feas_tol,
        rel_opt_tol=self.rel_opt_tol,
        rel_feas_tol=self.rel_feas_tol,
        smooth_evals=1,
        grad_evals=1,
        prox_evals=1,
    )

    x_work0 = x_flat0 / d_x if is_equilibrated else x_flat0
    mult_work0 = (cost_scale * (mult0 / d_y)) if is_equilibrated else mult0
    prox_mult_work0 = (
        (cost_scale * (d_x * prox_mult0)) if is_equilibrated else prox_mult0
    )
    prev_ec_work0 = (d_y * e_c0) if is_equilibrated else e_c0
    prev_ex_work0 = (e_x0 / d_x) if is_equilibrated else e_x0

    init_state = QPALMState(
        primal=x_work0,
        multiplier=mult_work0,
        prox_multiplier=prox_mult_work0,
        sigma_c=sigma_c0,
        sigma_x=sigma_x0,
        prev_e_c=prev_ec_work0,
        prev_e_x=prev_ex_work0,
        gamma=jnp.asarray(self.gamma_init, dtype=dtype),
        inner_tol=jnp.asarray(self.inner_tol_init, dtype=dtype),
        tracker=tracker0,
    )

    def _body(st: QPALMState) -> QPALMState:
      xk = st.primal
      yc = st.multiplier
      yx = st.prox_multiplier
      sc = st.sigma_c
      sx = st.sigma_x
      inv_gamma = 1.0 / st.gamma

      def _grad_from_cached(
          x_arr: jax.Array,
          gf: jax.Array,
          ax_val: jax.Array,
      ) -> jax.Array:
        zc = ax_val + yc / sc
        pc = sc * (zc - _proj_c_work(zc))
        g_c = a_mat.T @ pc
        g_tot = gf + inv_gamma * (x_arr - xk) + g_c
        if has_prox:
          zx = x_arr + yx / sx
          px = sx * (zx - _proj_x_work(zx))
          g_tot = g_tot + px
        return g_tot

      def _phi_grad(x_arr: jax.Array) -> jax.Array:
        gf = (
            (h_f_const @ x_arr + q_lin)
            if (h_f_const is not None and not a_is_bcoo)
            else _f_grad_work(x_arr)
        )
        if a_mat is not None:
          return _grad_from_cached(x_arr, gf, a_mat @ x_arr)
        x_pt = unflatten(x_arr)
        zc = tree.add(bound_c.apply(x_pt), jax.tree.map(jnp.divide, yc, sc))
        pc = jax.tree.map(
            jnp.multiply, sc, tree.sub(zc, _project_bound_c(bound_c, zc))
        )
        g_c, _ = _to_flat_array(bound_c.adjoint(pc))
        g_tot = gf + inv_gamma * (x_arr - xk) + g_c
        if has_prox:
          zx = x_arr + yx / sx
          px = sx * (zx - _proj_x(zx))
          g_tot = g_tot + px
        return g_tot

      ax_new_cached = None
      g_fx_new = None
      if use_woodbury and h_f_const is not None and a_dense is not None:
        u_all = (
            jnp.concatenate([a_dense, jnp.eye(n, dtype=dtype)], axis=0)
            if has_prox
            else a_dense
        )
        total_rows = int(u_all.shape[0])
        k_max = min(self.woodbury_max_rank, total_rows)
        init_w_all = jnp.zeros((total_rows,), dtype=dtype)
        init_fac = jnp.zeros((n, n), dtype=dtype)

        def _wb_cond(ist: _WoodburyInnerState) -> jax.Array:
          _, g_cur, k_in, _, _, _ = ist
          gnorm = (
              jnp.linalg.norm(g_cur, ord=jnp.inf)
              if self.norm_kind == "linf"
              else jnp.linalg.norm(g_cur)
          )
          return jnp.logical_and(
              k_in < self.max_inner_iters, gnorm > st.inner_tol
          )

        def _wb_body(ist: _WoodburyInnerState) -> _WoodburyInnerState:
          x_cur, g_cur, k_in, prev_w, prev_fac, has_fac = ist
          ax_cur = a_dense @ x_cur
          zc_cur = ax_cur + yc / sc
          _, j_c = jax.jvp(
              lambda w: w - _project_bound_c(bound_c, w),
              (zc_cur,),
              (jnp.ones_like(zc_cur),),
          )
          w_c = sc * j_c
          if has_prox:
            zx_cur = x_cur + yx / sx
            _, j_x = jax.jvp(
                lambda w: w - _proj_x(w),
                (zx_cur,),
                (jnp.ones_like(zx_cur),),
            )
            w_x = sx * j_x
            w_all = jnp.concatenate([w_c, w_x], axis=0)
          else:
            w_x = jnp.zeros((n,), dtype=dtype)
            w_all = w_c

          dw = w_all - prev_w
          n_diff = jnp.sum(jnp.abs(dw) > 0.0)
          can_wb = jnp.logical_and(has_fac, n_diff <= k_max)

          def _solve_wb(_: None) -> _WoodburyStep:
            _, top_idx = jax.lax.top_k(jnp.abs(dw), k_max)
            dw_top = dw[top_idx]
            u_mat = u_all[top_idx, :].T  # (n, k_max)
            z_mat = jax.scipy.linalg.cho_solve((prev_fac, True), u_mat)
            z0 = jax.scipy.linalg.cho_solve((prev_fac, True), -g_cur)
            # (H_old + U diag(dw) U^T)^-1 (-g) via Woodbury identity
            ut_z = u_mat.T @ z_mat
            cap = jnp.eye(k_max, dtype=dtype) + dw_top[:, None] * ut_z
            rhs_k = dw_top * (u_mat.T @ z0)
            y_k = jnp.linalg.solve(cap, rhs_k)
            d_wb = z0 - z_mat @ y_k
            return d_wb, prev_w, prev_fac

          def _solve_full(_: None) -> _WoodburyStep:
            h_mat = h_f_const + (a_dense.T * w_c) @ a_dense
            diag_add = inv_gamma + (w_x if has_prox else 0.0)
            idx = jnp.arange(n)
            h_mat = h_mat.at[idx, idx].add(diag_add)
            fac, _ = jax.scipy.linalg.cho_factor(h_mat, lower=True)
            d_full = jax.scipy.linalg.cho_solve((fac, True), -g_cur)
            return d_full, w_all, fac

          d, next_w, next_fac = jax.lax.cond(
              can_wb, _solve_wb, _solve_full, None
          )

          g_fx = h_f_const @ x_cur + q_lin
          g_fd = h_f_const @ d
          ad = a_dense @ d
          d_gfx = jnp.dot(d, g_fx)
          d_gfd = jnp.dot(d, g_fd)
          d_dxk = jnp.dot(d, x_cur - xk)
          d_dd = jnp.dot(d, d)
          zc0 = ax_cur + yc / sc

          def _psi_prime_quad(tau: jax.Array) -> jax.Array:
            val = d_gfx + tau * d_gfd + inv_gamma * (d_dxk + tau * d_dd)
            zc_t = zc0 + tau * ad
            pc_t = sc * (zc_t - _project_bound_c(bound_c, zc_t))
            val = val + jnp.dot(ad, pc_t)
            if has_prox:
              zx_t = (x_cur + tau * d) + yx / sx
              px_t = sx * (zx_t - _proj_x(zx_t))
              val = val + jnp.dot(d, px_t)
            return val

          psi0 = jnp.dot(g_cur, d)
          tau = _exact_1d_linesearch_cached(_psi_prime_quad, psi0, dtype=dtype)
          x_next = x_cur + tau * d
          g_next = _phi_grad(x_next)
          return (x_next, g_next, k_in + 1, next_w, next_fac, jnp.asarray(True))

        g_init = _phi_grad(xk)
        x_new, _, in_iters, _, _, _ = jax.lax.while_loop(
            _wb_cond,
            _wb_body,
            (
                xk,
                g_init,
                jnp.asarray(0, dtype=jnp.int32),
                init_w_all,
                init_fac,
                jnp.asarray(False),
            ),
        )
      elif h_f_const is not None and a_mat is not None:
        def _inner_cond_q(ist: _QuadInnerState) -> jax.Array:
          _, _, _, g_cur, k_in = ist
          g_chk = (g_cur / (cost_scale * d_x)) if is_equilibrated else g_cur
          gnorm = (
              jnp.linalg.norm(g_chk, ord=jnp.inf)
              if self.norm_kind == "linf"
              else jnp.linalg.norm(g_chk)
          )
          return jnp.logical_and(
              k_in < self.max_inner_iters,
              gnorm > st.inner_tol,
          )

        def _inner_body_q(ist: _QuadInnerState) -> _QuadInnerState:
          x_cur, g_fx, ax_cur, g_cur, k_in = ist
          zc_cur = ax_cur + yc / sc
          _, j_c = jax.jvp(
              lambda w: w - _proj_c_work(w),
              (zc_cur,),
              (jnp.ones_like(zc_cur),),
          )
          w_c = sc * j_c
          diag_add = jnp.full((n,), inv_gamma, dtype=dtype)
          if has_prox:
            zx_cur = x_cur + yx / sx
            _, j_x = jax.jvp(
                lambda w: w - _proj_x_work(w),
                (zx_cur,),
                (jnp.ones_like(zx_cur),),
            )
            diag_add = diag_add + sx * j_x
          if use_qdldl_ffi:
            if a_is_bcoo:
              d = _sparse_ldl_ffi.qdldl_kkt_solve_bcoo(
                  h_f_const,
                  a_mat.data,
                  a_mat.indices,
                  w_c,
                  diag_add,
                  -g_cur,
              )
            else:
              d = _sparse_ldl_ffi.qdldl_kkt_solve(
                  h_f_const,
                  a_dense,
                  w_c,
                  diag_add,
                  -g_cur,
              )
          elif _sparse_solve_cb is not None:
            d = jax.pure_callback(
                _sparse_solve_cb,
                jax.ShapeDtypeStruct((n,), dtype),
                h_f_const,
                a_dense,
                w_c,
                diag_add,
                -g_cur,
            )
          else:
            h_mat = h_f_const + (a_dense.T * w_c) @ a_dense
            idx = jnp.arange(n)
            h_mat = h_mat.at[idx, idx].add(diag_add)
            fac, lower = jax.scipy.linalg.cho_factor(h_mat, lower=True)
            d = jax.scipy.linalg.cho_solve((fac, lower), -g_cur)

          psi0 = jnp.dot(g_cur, d)
          if a_is_bcoo:
            _, g_fd = jax.jvp(_f_grad_work, (x_cur,), (d,))
          else:
            g_fd = h_f_const @ d
          ad = a_mat @ d

          d_gfx = jnp.dot(d, g_fx)
          d_gfd = jnp.dot(d, g_fd)
          d_dxk = jnp.dot(d, x_cur - xk)
          d_dd = jnp.dot(d, d)
          zc0 = zc_cur

          def _psi_prime_quad(tau: jax.Array) -> jax.Array:
            val = d_gfx + tau * d_gfd + inv_gamma * (d_dxk + tau * d_dd)
            zc_t = zc0 + tau * ad
            pc_t = sc * (zc_t - _proj_c_work(zc_t))
            val = val + jnp.dot(ad, pc_t)
            if has_prox:
              zx_t = (x_cur + tau * d) + yx / sx
              px_t = sx * (zx_t - _proj_x_work(zx_t))
              val = val + jnp.dot(d, px_t)
            return val

          tau = _exact_1d_linesearch_cached(
              _psi_prime_quad, psi0, dtype=dtype
          )
          x_next = x_cur + tau * d
          g_fx_next = g_fx + tau * g_fd
          ax_next = ax_cur + tau * ad
          g_next = _grad_from_cached(x_next, g_fx_next, ax_next)
          return (x_next, g_fx_next, ax_next, g_next, k_in + 1)

        g_fx0 = _f_grad_work(xk) if a_is_bcoo else (h_f_const @ xk + q_lin)
        ax0 = a_mat @ xk
        g_init = _grad_from_cached(xk, g_fx0, ax0)
        x_new, g_fx_new, ax_new_cached, _, in_iters = jax.lax.while_loop(
            _inner_cond_q,
            _inner_body_q,
            (xk, g_fx0, ax0, g_init, jnp.asarray(0, dtype=jnp.int32)),
        )
      else:
        def _inner_cond(ist: _InnerState) -> jax.Array:
          _, g_cur, k_in = ist
          gnorm = (
              jnp.linalg.norm(g_cur, ord=jnp.inf)
              if self.norm_kind == "linf"
              else jnp.linalg.norm(g_cur)
          )
          return jnp.logical_and(
              k_in < self.max_inner_iters,
              gnorm > st.inner_tol,
          )

        def _inner_body(ist: _InnerState) -> _InnerState:
          x_cur, g_cur, k_in = ist

          if False:
            pass
          else:
            def _hvp(v: jax.Array) -> jax.Array:
              _, hv = jax.jvp(_phi_grad, (x_cur,), (v,))
              return hv

            if use_chol:
              h_mat = jax.vmap(_hvp)(jnp.eye(n, dtype=dtype))
              h_mat = 0.5 * (h_mat + h_mat.T)
              fac, lower = jax.scipy.linalg.cho_factor(h_mat, lower=True)
              d = jax.scipy.linalg.cho_solve((fac, lower), -g_cur)
            else:
              x_pt_cur = unflatten(x_cur)
              zc_cur = tree.add(
                  bound_c.apply(x_pt_cur), jax.tree.map(jnp.divide, yc, sc)
              )
              ones_zc = jax.tree.map(jnp.ones_like, zc_cur)
              _, j_c = jax.jvp(
                  lambda w: tree.sub(w, _project_bound_c(bound_c, w)),
                  (zc_cur,),
                  (ones_zc,),
              )
              w_c = jax.tree.map(jnp.multiply, sc, j_c)
              diag_h = _diag_at_w_a(bound_c, w_c, n, dtype) + inv_gamma
              if has_prox:
                zx_cur = x_cur + yx / sx
                _, j_x = jax.jvp(
                    lambda w: w - _proj_x(w),
                    (zx_cur,),
                    (jnp.ones_like(zx_cur),),
                )
                diag_h = diag_h + sx * j_x
              inv_diag = 1.0 / jnp.maximum(
                  diag_h, jnp.asarray(1e-8, dtype=dtype)
              )
              d, _ = jax.scipy.sparse.linalg.cg(
                  _hvp,
                  -g_cur,
                  M=lambda v: inv_diag * v,
                  tol=self.cg_tol,
                  maxiter=self.cg_maxiter,
              )

          psi0 = jnp.dot(g_cur, d)
          if self.assume_quadratic:
            if h_f_const is not None and a_dense is not None:
              g_fx = h_f_const @ x_cur + q_lin
              g_fd = h_f_const @ d
              ad = a_dense @ d
            else:
              g_fx = _f_grad(x_cur)
              _, g_fd = jax.jvp(_f_grad, (x_cur,), (d,))
              ax_cur = bound_c.apply(unflatten(x_cur))
              ad = bound_c.apply(unflatten(d))

            d_gfx = jnp.dot(d, g_fx)
            d_gfd = jnp.dot(d, g_fd)
            d_dxk = jnp.dot(d, x_cur - xk)
            d_dd = jnp.dot(d, d)
            zc0 = tree.add(ax_cur, jax.tree.map(jnp.divide, yc, sc))

            def _psi_prime_quad(tau: jax.Array) -> jax.Array:
              val = d_gfx + tau * d_gfd + inv_gamma * (d_dxk + tau * d_dd)
              zc_t = tree.add_scaled(zc0, tau, ad)
              pc_t = jax.tree.map(
                  jnp.multiply,
                  sc,
                  tree.sub(zc_t, _project_bound_c(bound_c, zc_t)),
              )
              val = val + tree.vdot(ad, pc_t)
              if has_prox:
                zx_t = (x_cur + tau * d) + yx / sx
                px_t = sx * (zx_t - _proj_x(zx_t))
                val = val + jnp.dot(d, px_t)
              return val

            tau = _exact_1d_linesearch_cached(
                _psi_prime_quad, psi0, dtype=dtype
            )
          else:
            tau = _exact_1d_linesearch_cached(
                lambda t: jnp.dot(_phi_grad(x_cur + t * d), d),
                psi0,
                dtype=dtype,
            )

          x_next = x_cur + tau * d
          g_next = _phi_grad(x_next)
          return (x_next, g_next, k_in + 1)

        g_init = _phi_grad(xk)
        x_new, _, in_iters = jax.lax.while_loop(
            _inner_cond,
            _inner_body,
            (xk, g_init, jnp.asarray(0, dtype=jnp.int32)),
        )

      # Outer dual update (in working/scaled space)
      use_q_cached = (not use_woodbury) and (
          h_f_const is not None and a_mat is not None
      )
      x_new_pt = unflatten(x_new)
      ax_new = ax_new_cached if use_q_cached else bound_c.apply(x_new_pt)
      zc_new = tree.add(ax_new, jax.tree.map(jnp.divide, yc, sc))
      proj_zc = _proj_c_work(zc_new)
      yc_new = jax.tree.map(jnp.multiply, sc, tree.sub(zc_new, proj_zc))

      if has_prox:
        zx_new = x_new + yx / sx
        yx_new = sx * (zx_new - _proj_x_work(zx_new))
      else:
        yx_new = yx

      # Residuals in original problem space
      e_c_work = tree.sub(ax_new, _proj_c_work(ax_new))
      e_x_work = (
          x_new - _proj_x_work(x_new) if has_prox else jnp.zeros_like(x_new)
      )

      if is_equilibrated:
        x_orig = d_x * x_new
        x_orig_pt = unflatten(x_orig)
        yc_orig = (yc_new * d_y) / cost_scale
        e_c_orig = e_c_work / d_y
        e_x_orig = d_x * e_x_work
        cand_obj, _ = _f_val_and_grad(x_orig)
        aty_work = a_mat.T @ yc_new
        v_orig = (g_fx_new + aty_work + yx_new) / (cost_scale * d_x)
      else:
        x_orig = x_new
        x_orig_pt = x_new_pt
        yc_orig = yc_new
        e_c_orig = e_c_work
        e_x_orig = e_x_work
        cand_obj, gf_new = _f_val_and_grad(x_new)
        if use_q_cached:
          aty, _ = _to_flat_array(a_mat.T @ yc_new)
        else:
          aty, _ = _to_flat_array(bound_c.adjoint(yc_new))
        v_orig = gf_new + aty + yx_new

      cand_opt = (
          jnp.linalg.norm(v_orig, ord=jnp.inf)
          if self.norm_kind == "linf"
          else jnp.linalg.norm(v_orig)
      )
      cand_feas = jnp.maximum(
          tree.norm(e_c_orig, kind=self.norm_kind),
          tree.norm(e_x_orig, kind=self.norm_kind),
      )

      # Per-constraint penalty update (QPALM rule, checking unscaled feas_tol)
      def _upd_sigma(
          s_leaf: jax.Array,
          e_leaf: jax.Array,
          pe_leaf: jax.Array,
          e_orig_leaf: jax.Array,
      ) -> jax.Array:
        grew = jnp.logical_and(
            jnp.abs(e_leaf) > self.theta_feas * jnp.abs(pe_leaf),
            jnp.abs(e_orig_leaf) > self.feas_tol,
        )
        return jnp.where(
            grew,
            jnp.minimum(s_leaf * self.sigma_factor, self.sigma_max),
            s_leaf,
        )

      sc_next = jax.tree.map(_upd_sigma, sc, e_c_work, st.prev_e_c, e_c_orig)
      sx_next = (
          _upd_sigma(sx, e_x_work, st.prev_e_x, e_x_orig) if has_prox else sx
      )
      gamma_next = jnp.minimum(st.gamma * self.gamma_factor, self.gamma_max)
      inner_tol_next = jnp.maximum(
          jnp.asarray(0.5 * self.opt_tol, dtype=dtype),
          jnp.minimum(
              st.inner_tol * self.inner_tol_factor,
              jnp.asarray(0.1, dtype=dtype) * jnp.maximum(cand_opt, cand_feas),
          ),
      )

      cand_finite = jnp.logical_and(
          jnp.all(jnp.isfinite(x_orig)),
          jnp.logical_and(jnp.isfinite(cand_opt), jnp.isfinite(cand_feas)),
      )
      tracker = st.tracker.update_incumbent(
          x_orig_pt,
          yc_orig,
          unflatten(v_orig),
          cand_opt,
          cand_feas,
          cand_obj,
      ).advance(
          inner_inc=in_iters,
          smooth_inc=in_iters + 1,
          grad_inc=in_iters + 2,
          prox_inc=in_iters + 1,
          max_iters=self.max_alm_iters,
          finite_ok=cand_finite,
      )

      return QPALMState(
          primal=x_new,
          multiplier=yc_new,
          prox_multiplier=yx_new,
          sigma_c=sc_next,
          sigma_x=sx_next,
          prev_e_c=e_c_work,
          prev_e_x=e_x_work,
          gamma=gamma_next,
          inner_tol=inner_tol_next,
          tracker=tracker,
      )

    return base.run_solver_loop(
        _body,
        init_state,
        jit=jit,
        callback=callback,
        time_limit=time_limit,
    )

  def solve(
      self,
      problem: problem_mod.CompositeProblem,
      x0: _typing.PyTree,
      *,
      jit: bool = True,
      callback: _typing.CallbackFn[QPALMState] | None = None,
      time_limit: float | None = None,
  ) -> base.Solution:
    """Solves a constrained `CompositeProblem` using `QPALM`."""
    return self.solve_state(
        problem,
        x0,
        jit=jit,
        callback=callback,
        time_limit=time_limit,
    ).to_solution()
