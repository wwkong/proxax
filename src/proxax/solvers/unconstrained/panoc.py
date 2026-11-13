"""PANOC: Proximal Averaged Newton-type Method for Optimal Control."""

from __future__ import annotations

import dataclasses
import typing

import jax
import jax.numpy as jnp

from proxax import _loop
from proxax import _typing
from proxax import certificates
from proxax import problem as problem_mod
from proxax import tree
from proxax.oracle import composite
from proxax.solvers import base


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class PANOCState(base.SolverTrackerMixin):
  """Iteration state of the `PANOC` composite quasi-Newton solver."""

  x: _typing.PyTree
  lipschitz: jax.Array
  s_hist: _typing.PyTree
  y_hist: _typing.PyTree
  rho_hist: jax.Array
  hist_head: jax.Array
  hist_count: jax.Array
  tracker: base.SolverTracker


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _LCheckCarry:
  """Internal carry for smoothness backtracking in `PANOC`."""

  lipschitz: jax.Array
  x_bar: _typing.PyTree
  f_bar: jax.Array
  grad_bar: _typing.PyTree
  h_bar: jax.Array
  accepted: jax.Array
  evals: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _FBEBacktrackCarry:
  """Internal carry for PANOC FBE line search."""

  tau: jax.Array
  x_next: _typing.PyTree
  fbe_next: jax.Array
  accepted: jax.Array
  steps: jax.Array
  evals: jax.Array


def _tree_slice(buf: _typing.PyTree, idx: jax.Array) -> _typing.PyTree:
  return jax.tree.map(lambda a: a[idx], buf)


def _tree_set(
    buf: _typing.PyTree, idx: jax.Array, val: _typing.PyTree
) -> _typing.PyTree:
  return jax.tree.map(lambda a, v: a.at[idx].set(v), buf, val)


def _lbfgs_direction(
    r: _typing.PyTree,
    s_hist: _typing.PyTree,
    y_hist: _typing.PyTree,
    rho_hist: jax.Array,
    hist_head: jax.Array,
    hist_count: jax.Array,
    memory: int,
    dtype: jnp.dtype,
) -> _typing.PyTree:
  """Computes `-H_k r` via two-loop L-BFGS recursion on PyTree buffers."""
  q = r
  alphas = jnp.zeros(memory, dtype=dtype)

  def _bwd_body(
      j: jax.Array, carry: tuple[_typing.PyTree, jax.Array]
  ) -> tuple[_typing.PyTree, jax.Array]:
    q_cur, al_cur = carry
    idx = (hist_head - 1 - j) % memory
    s_i = _tree_slice(s_hist, idx)
    y_i = _tree_slice(y_hist, idx)
    alpha_i = rho_hist[idx] * tree.vdot(s_i, q_cur)
    q_next = tree.add_scaled(q_cur, -alpha_i, y_i)
    return q_next, al_cur.at[idx].set(alpha_i)

  q, alphas = jax.lax.fori_loop(0, hist_count, _bwd_body, (q, alphas))

  def _scale_q(q_in: _typing.PyTree) -> _typing.PyTree:
    last_idx = (hist_head - 1) % memory
    s_last = _tree_slice(s_hist, last_idx)
    y_last = _tree_slice(y_hist, last_idx)
    yy = tree.norm_sq(y_last)
    sy = tree.vdot(s_last, y_last)
    gamma_h = jnp.where(yy > 0.0, sy / jnp.maximum(yy, 1e-30), 1.0)
    return tree.scale(gamma_h, q_in)

  z = jax.lax.cond(hist_count > 0, _scale_q, lambda x: x, q)

  def _fwd_body(i: jax.Array, z_cur: _typing.PyTree) -> _typing.PyTree:
    idx = (hist_head - hist_count + i) % memory
    s_i = _tree_slice(s_hist, idx)
    y_i = _tree_slice(y_hist, idx)
    beta_i = rho_hist[idx] * tree.vdot(y_i, z_cur)
    return tree.add_scaled(z_cur, alphas[idx] - beta_i, s_i)

  z = jax.lax.fori_loop(0, hist_count, _fwd_body, z)
  return tree.scale(-1.0, z)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class PANOC:
  """Composite Quasi-Newton (`PANOC`) solver (Stella, Themelis, Patrinos 2017).

  Combines proximal gradient steps with L-BFGS directions over the smooth
  Forward-Backward Envelope (FBE), achieving superlinear local convergence on
  nonconvex or ill-conditioned composite problems `min_x f(x) + h(x)`.

  Attributes:
    opt_tol: Absolute stationarity tolerance `rho > 0`.
    rel_opt_tol: Optional relative stationarity tolerance vs step-0 norm.
    norm_kind: Residual norm (`'l2'` or `'linf'`).
    memory: Number of L-BFGS `(s, y)` secant pairs to retain.
    lipschitz_init: Initial smoothness estimate `L_0 > 0`.
    lipschitz_max: Maximum smoothness cap during backtracking.
    beta: Stepsize safety factor `gamma_k = beta / L_k` in `(0, 1)`.
    sigma: FBE sufficient-decrease factor in `(0, 1)`.
    gamma_up: Multiplicative smoothness growth factor (`> 1`).
    max_iters: Maximum outer PANOC iterations.
    max_backtracks: Maximum smoothness doublings per outer iteration.
    max_ls_steps: Maximum FBE backtracking halvings per outer iteration.
  """

  opt_tol: float = 1e-5
  rel_opt_tol: float = 0.0
  norm_kind: str = "l2"
  memory: int = 10
  lipschitz_init: float = 1.0
  lipschitz_max: float = 1e12
  beta: float = 0.95
  sigma: float = 0.5
  gamma_up: float = 2.0
  max_iters: int = 2000
  max_backtracks: int = 30
  max_ls_steps: int = 15

  def __post_init__(self) -> None:
    """Validates `PANOC` hyperparameters."""
    if self.opt_tol <= 0.0:
      raise ValueError("`opt_tol` must be positive.")
    if self.memory < 1:
      raise ValueError("`memory` must be >= 1.")
    if not (0.0 < self.beta < 1.0):
      raise ValueError("`beta` must lie in (0, 1).")

  def solve(
      self,
      problem: problem_mod.CompositeProblem,
      x0: _typing.PyTree,
      *,
      jit: bool = True,
      callback: typing.Any = None,
      time_limit: float | None = None,
  ) -> base.Solution:
    """Solves an unconstrained `CompositeProblem` using `PANOC`."""
    if problem.is_constrained:
      raise ValueError("`PANOC` solves unconstrained problems only.")
    return self.solve_oracle(
        problem.oracle,
        x0,
        jit=jit,
        callback=callback,
        time_limit=time_limit,
    ).to_solution()

  def solve_oracle(
      self,
      oracle: composite.CompositeOracleProtocol,
      x0: _typing.PyTree,
      *,
      lipschitz_init: jax.Array | float | None = None,
      opt_tol_override: jax.Array | float | None = None,
      jit: bool = True,
      callback: typing.Any = None,
      time_limit: float | None = None,
  ) -> PANOCState:
    """Solves a `CompositeOracle` directly using `PANOC`."""
    dtype = tree.infer_dtype(x0)
    x_dom0 = oracle.domain_project(x0)
    if lipschitz_init is not None:
      L_seed = jnp.asarray(lipschitz_init, dtype=dtype)
    else:
      lip_guess = (
          float(oracle.curvature_upper)
          if oracle.curvature_upper is not None
          and float(oracle.curvature_upper) > 0.0
          else float(self.lipschitz_init)
      )
      L_seed = jnp.asarray(lip_guess, dtype=dtype)
    ref0 = certificates.refine(oracle, x_dom0, L_seed, jit=jit)
    L0 = ref0.lipschitz_used
    res0 = tree.norm(ref0.v_hat, kind=self.norm_kind)
    tracker0 = base.SolverTracker.from_init(
        x=ref0.x_hat,
        subgrad=ref0.v_hat,
        opt_res=res0,
        obj=ref0.obj_val,
        opt_tol=self.opt_tol,
        rel_opt_tol=self.rel_opt_tol,
        opt_tol_override=opt_tol_override,
        smooth_evals=ref0.smooth_evals,
        grad_evals=ref0.grad_evals,
        prox_evals=ref0.prox_evals,
    )

    s_hist0 = jax.tree.map(
        lambda a: jnp.zeros((self.memory, *a.shape), dtype=dtype), x_dom0
    )
    y_hist0 = jax.tree.map(
        lambda a: jnp.zeros((self.memory, *a.shape), dtype=dtype), x_dom0
    )
    rho_hist0 = jnp.zeros(self.memory, dtype=dtype)
    init_state = PANOCState(
        x=ref0.x_hat,
        lipschitz=L0,
        s_hist=s_hist0,
        y_hist=y_hist0,
        rho_hist=rho_hist0,
        hist_head=jnp.asarray(0, dtype=jnp.int32),
        hist_count=jnp.asarray(0, dtype=jnp.int32),
        tracker=tracker0,
    )

    def _fbe_at(
        x_pt: _typing.PyTree, gamma: jax.Array
    ) -> tuple[
        _typing.PyTree,
        jax.Array,
        _typing.PyTree,
        jax.Array,
        jax.Array,
    ]:
      f_val, grad = oracle.smooth_value_and_grad(x_pt)
      x_bar = oracle.prox(tree.add_scaled(x_pt, -gamma, grad), gamma)
      h_bar = oracle.proximal_value(x_bar)
      r = tree.sub(x_pt, x_bar)
      fbe = (
          f_val
          + h_bar
          - tree.vdot(grad, r)
          + (0.5 / gamma) * tree.norm_sq(r)
      )
      return x_bar, f_val, grad, h_bar, fbe

    def _body(st: PANOCState) -> PANOCState:
      eps_val = tree.eps(dtype)
      f_cur, grad_cur = oracle.smooth_value_and_grad(st.x)

      def _trial_l(
          L_val: jax.Array,
      ) -> tuple[
          _typing.PyTree, jax.Array, _typing.PyTree, jax.Array, jax.Array
      ]:
        gamma = self.beta / L_val
        x_bar = oracle.prox(tree.add_scaled(st.x, -gamma, grad_cur), gamma)
        f_bar, grad_bar = oracle.smooth_value_and_grad(x_bar)
        h_bar = oracle.proximal_value(x_bar)
        diff = tree.sub(x_bar, st.x)
        quad_ub = (
            f_cur
            + tree.vdot(grad_cur, diff)
            + 0.5 * self.beta * L_val * tree.norm_sq(diff)
        )
        slack = 10.0 * eps_val * (1.0 + jnp.abs(f_cur))
        ok = f_bar <= quad_ub + slack
        return x_bar, f_bar, grad_bar, h_bar, jnp.asarray(ok, dtype=jnp.bool_)

      xb0, fb0, gb0, hb0, ok0 = _trial_l(st.lipschitz)
      ls_init = _LCheckCarry(
          lipschitz=st.lipschitz,
          x_bar=xb0,
          f_bar=fb0,
          grad_bar=gb0,
          h_bar=hb0,
          accepted=ok0,
          evals=jnp.asarray(1, dtype=jnp.int32),
      )

      def _l_cond(c: _LCheckCarry) -> jax.Array:
        return jnp.logical_and(
            jnp.logical_not(c.accepted),
            jnp.logical_and(
                c.evals <= self.max_backtracks,
                c.lipschitz < self.lipschitz_max,
            ),
        )

      def _l_body(c: _LCheckCarry) -> _LCheckCarry:
        next_L = jnp.minimum(
            c.lipschitz * self.gamma_up,
            jnp.asarray(self.lipschitz_max, dtype=dtype),
        )
        xb, fb, gb, hb, ok = _trial_l(next_L)
        return _LCheckCarry(
            lipschitz=next_L,
            x_bar=xb,
            f_bar=fb,
            grad_bar=gb,
            h_bar=hb,
            accepted=ok,
            evals=c.evals + 1,
        )

      l_fin = _loop.while_loop(_l_cond, _l_body, ls_init, jit=jit)
      L_used = l_fin.lipschitz
      gamma = self.beta / L_used
      x_bar = l_fin.x_bar
      f_bar = l_fin.f_bar
      grad_bar = l_fin.grad_bar
      h_bar = l_fin.h_bar

      r_cur = tree.sub(st.x, x_bar)
      fbe_cur = (
          f_cur
          + h_bar
          - tree.vdot(grad_cur, r_cur)
          + (0.5 / gamma) * tree.norm_sq(r_cur)
      )

      # Exact composite subgradient at x_bar.
      v_bar = tree.add(
          tree.scale(1.0 / gamma, r_cur),
          tree.sub(grad_bar, grad_cur),
      )
      res_bar = tree.norm(v_bar, kind=self.norm_kind)
      obj_bar = f_bar + h_bar

      lip_changed = L_used > st.lipschitz * (1.0 + 1e-6)
      count_cur = jnp.where(lip_changed, 0, st.hist_count)

      d_cur = _lbfgs_direction(
          r_cur,
          st.s_hist,
          st.y_hist,
          st.rho_hist,
          st.hist_head,
          count_cur,
          self.memory,
          dtype,
      )

      # PANOC backtracking on FBE along (1 - tau) * x_bar + tau * (x + d_cur).
      r_norm_sq = tree.norm_sq(r_cur)
      suff_dec = (
          self.sigma * (1.0 - self.beta) / (2.0 * gamma)
      ) * r_norm_sq
      target_fbe = (
          fbe_cur - suff_dec + 10.0 * eps_val * (1.0 + jnp.abs(fbe_cur))
      )
      dir_plus_r = tree.add(d_cur, r_cur)

      x_try0 = tree.add(x_bar, dir_plus_r)
      _, _, _, _, fbe_try0 = _fbe_at(x_try0, gamma)
      fbe_init = _FBEBacktrackCarry(
          tau=jnp.asarray(1.0, dtype=dtype),
          x_next=x_try0,
          fbe_next=fbe_try0,
          accepted=jnp.asarray(fbe_try0 <= target_fbe, dtype=jnp.bool_),
          steps=jnp.asarray(0, dtype=jnp.int32),
          evals=jnp.asarray(1, dtype=jnp.int32),
      )

      def _fbe_cond(c: _FBEBacktrackCarry) -> jax.Array:
        return jnp.logical_and(
            jnp.logical_not(c.accepted),
            c.steps < self.max_ls_steps,
        )

      def _fbe_body(c: _FBEBacktrackCarry) -> _FBEBacktrackCarry:
        tau_new = 0.5 * c.tau
        x_try = tree.add_scaled(x_bar, tau_new, dir_plus_r)
        _, _, _, _, fbe_try = _fbe_at(x_try, gamma)
        return _FBEBacktrackCarry(
            tau=tau_new,
            x_next=x_try,
            fbe_next=fbe_try,
            accepted=jnp.asarray(fbe_try <= target_fbe, dtype=jnp.bool_),
            steps=c.steps + 1,
            evals=c.evals + 1,
        )

      fbe_fin = _loop.while_loop(_fbe_cond, _fbe_body, fbe_init, jit=jit)
      x_next = jax.lax.cond(
          fbe_fin.accepted, lambda: fbe_fin.x_next, lambda: x_bar
      )

      # Compute r_next for L-BFGS secant update.
      xb_next, _, _, _, _ = _fbe_at(x_next, gamma)
      r_next = tree.sub(x_next, xb_next)
      s_k = tree.sub(x_next, st.x)
      y_k = tree.sub(r_next, r_cur)
      sy = tree.vdot(s_k, y_k)
      s_norm = tree.norm(s_k)
      y_norm = tree.norm(y_k)
      valid_secant = sy > 1e-8 * s_norm * y_norm

      _SecantBufs = tuple[
          _typing.PyTree, _typing.PyTree, jax.Array, jax.Array, jax.Array
      ]

      def _push_secant(bufs: _SecantBufs) -> _SecantBufs:
        sh, yh, rh, hd, cnt = bufs
        rho_val = 1.0 / jnp.maximum(sy, 1e-30)
        sh = _tree_set(sh, hd, s_k)
        yh = _tree_set(yh, hd, y_k)
        rh = rh.at[hd].set(rho_val)
        return (
            sh,
            yh,
            rh,
            (hd + 1) % self.memory,
            jnp.minimum(cnt + 1, self.memory),
        )

      s_hist_n, y_hist_n, rho_hist_n, head_n, count_n = jax.lax.cond(
          valid_secant,
          _push_secant,
          lambda b: b,
          (st.s_hist, st.y_hist, st.rho_hist, st.hist_head, count_cur),
      )

      total_evals = l_fin.evals + fbe_fin.evals + 1
      next_tracker = st.tracker.update_incumbent(
          x_bar, v_bar, res_bar, obj_bar
      ).advance(
          smooth_inc=1 + total_evals,
          grad_inc=1 + total_evals,
          prox_inc=total_evals,
          max_iters=self.max_iters,
      )
      return PANOCState(
          x=x_next,
          lipschitz=L_used,
          s_hist=s_hist_n,
          y_hist=y_hist_n,
          rho_hist=rho_hist_n,
          hist_head=head_n,
          hist_count=count_n,
          tracker=next_tracker,
      )

    return typing.cast(
        PANOCState,
        base.run_solver_loop(
            _body,
            init_state,
            jit=jit,
            callback=callback,
            time_limit=time_limit,
        ),
    )
