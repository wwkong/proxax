"""Nonconvex FISTA (`NC-FISTA`) solver (Liang, Monteiro, & Sim 2021)."""

from __future__ import annotations

import dataclasses

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
class NCFISTAState:
  """Iteration state of the `NCFISTA` solver."""

  x: _typing.PyTree
  y: _typing.PyTree
  a_sum: jax.Array
  lam: jax.Array
  xi: jax.Array
  best_x: _typing.PyTree
  best_subgrad: _typing.PyTree
  best_opt_res: jax.Array
  best_obj: jax.Array
  target_opt_tol: jax.Array
  outer_iters: jax.Array
  inner_iters: jax.Array
  smooth_evals: jax.Array
  grad_evals: jax.Array
  prox_evals: jax.Array
  status: jax.Array


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class NCFISTA:
  """Nonconvex Fast Iterative Shrinkage-Thresholding (`NC-FISTA`) solver.

  Implements `src/solvers/NC_FISTA.m` from `wwkong/nc_opt` based on J. Liang,
  R. D. C. Monteiro, and C. K. Sim, *"A FISTA-type accelerated gradient
  algorithm for solving smooth nonconvex composite optimization problems,"*
  Computational Optimization and Applications, 79(3):649--679, 2021.

  Uses fixed step size `lam` (default `0.99 / L` where `L = max(M, m)`) and
  damping parameter `xi` (`1.05 * m` when `m > 0`, reducing smoothly to standard
  convex FISTA `A_0 = 0, tau = 0` when `m = 0`). At every step, forms the exact
  composite subgradient `v = ((1 + tau) / lam) * (x_tilde - y_next) +
  grad_f(y_next) - grad_f(x_tilde)` in `grad_f(y_next) + dh(y_next)`.

  Attributes:
    opt_tol: Absolute stationarity tolerance `rho > 0`.
    rel_opt_tol: Optional relative stationarity tolerance vs step-0 norm.
    norm_kind: Residual norm (`'l2'` or `'linf'`).
    upper_curvature: Optional global upper smoothness constant `M > 0`.
    lower_curvature: Optional weak-convexity constant `m >= 0`.
    lam: Optional step size `lambda > 0` (defaults to `0.99 / max(M, m)`).
    xi: Optional nonconvex damping constant `xi > m` (defaults to `1.05 * m`
      when `m > 0` and `0.0` when `m == 0`).
    lipschitz_init: Fallback smoothness seed when `M` is unknown.
    max_iters: Maximum `NC-FISTA` iterations.
  """

  opt_tol: float = 1e-5
  rel_opt_tol: float = 0.0
  norm_kind: str = "l2"
  upper_curvature: float | None = None
  lower_curvature: float | None = None
  lam: float | None = None
  xi: float | None = None
  lipschitz_init: float = 1.0
  max_iters: int = 2000

  def solve_oracle(
      self,
      oracle: composite.CompositeOracleProtocol,
      x0: _typing.PyTree,
      *,
      opt_tol_override: _typing.Scalar | None = None,
      jit: bool = True,
      callback: _typing.CallbackFn[NCFISTAState] | None = None,
      time_limit: float | None = None,
  ) -> NCFISTAState:
    """Runs `NCFISTA` directly on any `CompositeOracleProtocol`."""
    x_proj = oracle.domain_project(tree.as_float_tree(x0))
    dtype = tree.infer_dtype(x_proj)

    if self.lower_curvature is not None:
      m_val = jnp.asarray(self.lower_curvature, dtype=dtype)
    elif oracle.curvature_lower is not None:
      m_val = jnp.asarray(oracle.curvature_lower, dtype=dtype)
    else:
      m_val = jnp.asarray(1.0, dtype=dtype)
    m_val = jnp.maximum(m_val, jnp.asarray(0.0, dtype=dtype))

    known_M = (
        self.upper_curvature is not None or oracle.curvature_upper is not None
    )
    if self.upper_curvature is not None:
      M_seed = jnp.asarray(self.upper_curvature, dtype=dtype)
    elif oracle.curvature_upper is not None:
      M_seed = jnp.asarray(oracle.curvature_upper, dtype=dtype)
    else:
      M_seed = jnp.asarray(self.lipschitz_init, dtype=dtype)

    L_seed = jnp.maximum(
        jnp.maximum(M_seed, m_val),
        jnp.asarray(1e-8, dtype=dtype),
    )
    ref0 = certificates.refine(oracle, x_proj, L_seed, jit=jit)
    L_eff = L_seed if known_M else jnp.maximum(L_seed, ref0.lipschitz_used)

    lam_arr = (
        jnp.asarray(self.lam, dtype=dtype)
        if self.lam is not None
        else 0.99 / L_eff
    )
    if self.xi is not None:
      xi_arr = jnp.asarray(self.xi, dtype=dtype)
    else:
      xi_arr = 1.05 * m_val

    has_xi = xi_arr > m_val + jnp.asarray(1e-12, dtype=dtype)
    denom = jnp.maximum((xi_arr - m_val) ** 2, jnp.asarray(1e-24, dtype=dtype))
    a_sum0 = jnp.where(
        has_xi,
        2.0 * xi_arr * (xi_arr + m_val) / denom,
        jnp.asarray(0.0, dtype=dtype),
    )

    res0 = tree.norm(ref0.v_hat, kind=self.norm_kind)
    if opt_tol_override is not None:
      target_tol = jnp.asarray(opt_tol_override, dtype=dtype)
    else:
      target_tol = jnp.asarray(self.opt_tol, dtype=dtype) + (
          jnp.asarray(self.rel_opt_tol, dtype=dtype) * res0
      )

    finite0 = jnp.logical_and(tree.all_finite(ref0.x_hat), jnp.isfinite(res0))
    converged0 = res0 <= target_tol
    running_code = jnp.asarray(int(base.SolveStatus.RUNNING), dtype=jnp.int32)
    status0 = jnp.where(
        jnp.logical_not(finite0),
        jnp.asarray(int(base.SolveStatus.NON_FINITE), dtype=jnp.int32),
        jnp.where(
            converged0,
            jnp.asarray(int(base.SolveStatus.CONVERGED), dtype=jnp.int32),
            running_code,
        ),
    )

    init_state = NCFISTAState(
        x=ref0.x_hat,
        y=ref0.x_hat,
        a_sum=a_sum0,
        lam=lam_arr,
        xi=xi_arr,
        best_x=ref0.x_hat,
        best_subgrad=ref0.v_hat,
        best_opt_res=res0,
        best_obj=ref0.obj_val,
        target_opt_tol=target_tol,
        outer_iters=jnp.asarray(0, dtype=jnp.int32),
        inner_iters=jnp.asarray(0, dtype=jnp.int32),
        smooth_evals=ref0.smooth_evals,
        grad_evals=ref0.grad_evals,
        prox_evals=ref0.prox_evals,
        status=status0,
    )

    def _cond(st: NCFISTAState) -> jax.Array:
      return st.status == running_code

    def _body(st: NCFISTAState) -> NCFISTAState:
      a_step = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * st.a_sum))
      a_next = st.a_sum + a_step
      tau = 2.0 * st.xi * st.lam / a_step
      x_tilde = tree.convex_comb(st.a_sum, st.y, a_step, st.x)

      _, grad_tx = oracle.smooth_value_and_grad(x_tilde)
      eff_step = st.lam / (1.0 + tau)
      y_next = oracle.prox(
          tree.add_scaled(x_tilde, -eff_step, grad_tx),
          eff_step,
      )
      denom_x = a_step * (tau * a_step + 1.0)
      c_y_next = (1.0 + tau) * a_next / denom_x
      c_y_prev = st.a_sum / denom_x
      x_next = jax.tree.map(
          lambda yn, yp: c_y_next * yn - c_y_prev * yp,
          y_next,
          st.y,
      )

      f_yn, grad_yn = oracle.smooth_value_and_grad(y_next)
      v_cand = tree.add(
          tree.scale(1.0 / eff_step, tree.sub(x_tilde, y_next)),
          tree.sub(grad_yn, grad_tx),
      )
      res_cand = tree.norm(v_cand, kind=self.norm_kind)
      obj_yn = f_yn + oracle.proximal_value(y_next)

      cand_finite = jnp.logical_and(
          tree.all_finite(y_next),
          jnp.isfinite(res_cand),
      )
      is_better = jnp.logical_and(cand_finite, res_cand <= st.best_opt_res)
      best_x = jax.tree.map(
          lambda cy, bx: jnp.where(is_better, cy, bx),
          y_next,
          st.best_x,
      )
      best_v = jax.tree.map(
          lambda cv, bv: jnp.where(is_better, cv, bv),
          v_cand,
          st.best_subgrad,
      )
      best_res = jnp.where(is_better, res_cand, st.best_opt_res)
      best_obj = jnp.where(is_better, obj_yn, st.best_obj)

      next_outer = st.outer_iters + 1
      converged_now = best_res <= st.target_opt_tol
      next_status = jnp.where(
          jnp.logical_not(cand_finite),
          jnp.asarray(int(base.SolveStatus.NON_FINITE), dtype=jnp.int32),
          jnp.where(
              converged_now,
              jnp.asarray(int(base.SolveStatus.CONVERGED), dtype=jnp.int32),
              jnp.where(
                  next_outer >= self.max_iters,
                  jnp.asarray(
                      int(base.SolveStatus.MAX_ITER_REACHED), dtype=jnp.int32
                  ),
                  running_code,
              ),
          ),
      )

      return NCFISTAState(
          x=x_next,
          y=y_next,
          a_sum=a_next,
          lam=st.lam,
          xi=st.xi,
          best_x=best_x,
          best_subgrad=best_v,
          best_opt_res=best_res,
          best_obj=best_obj,
          target_opt_tol=st.target_opt_tol,
          outer_iters=next_outer,
          inner_iters=st.inner_iters + 1,
          smooth_evals=st.smooth_evals + 2,
          grad_evals=st.grad_evals + 2,
          prox_evals=st.prox_evals + 1,
          status=next_status,
      )

    def _on_timeout(st: NCFISTAState) -> NCFISTAState:
      return dataclasses.replace(
          st,
          status=jnp.asarray(
              int(base.SolveStatus.TIME_LIMIT_REACHED),
              dtype=jnp.int32,
          ),
      )

    return _loop.while_loop(
        _cond,
        _body,
        init_state,
        jit=jit,
        callback=callback,
        time_limit=time_limit,
        on_time_limit=_on_timeout,
    )

  def solve(
      self,
      problem: problem_mod.CompositeProblem,
      x0: _typing.PyTree,
      *,
      jit: bool = True,
      callback: _typing.CallbackFn[NCFISTAState] | None = None,
      time_limit: float | None = None,
  ) -> base.Solution:
    """Solves an unconstrained `CompositeProblem` using `NCFISTA`."""
    if problem.is_constrained:
      raise ValueError(
          "NCFISTA solves unconstrained composite problems; pass `QPAIPP`, "
          "`QPAPD`, `IALM`, or use `proxax.solve(problem, x0)`."
      )
    fin = self.solve_oracle(
        problem.oracle,
        x0,
        jit=jit,
        callback=callback,
        time_limit=time_limit,
    )
    stats = base.SolveStats(
        outer_iters=fin.outer_iters,
        inner_iters=fin.inner_iters,
        smooth_evals=fin.smooth_evals,
        grad_evals=fin.grad_evals,
        prox_evals=fin.prox_evals,
        opt_residual=fin.best_opt_res,
        feas_residual=jnp.asarray(0.0, dtype=fin.best_opt_res.dtype),
        obj_val=fin.best_obj,
    )
    return base.Solution(
        x=fin.best_x,
        dual=None,
        subgrad=fin.best_subgrad,
        status=fin.status,
        stats=stats,
    )
