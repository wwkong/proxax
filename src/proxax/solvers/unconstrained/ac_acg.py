"""Average Curvature ACG (`AC-ACG`) solver (Liang & Monteiro 2019)."""

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
class ACACGState:
  """Iteration state of the `ACACG` (`AC-ACG`) solver."""

  x: _typing.PyTree
  y: _typing.PyTree
  a_sum: jax.Array
  m_est: jax.Array
  m_floor: jax.Array
  c_sum: jax.Array
  good_steps: jax.Array
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
class ACACG:
  """Average Curvature Accelerated Composite Gradient (`AC-ACG`) solver.

  Implements `src/solvers/AC_ACG.m` from `wwkong/nc_opt` based on J. Liang and
  R. D. C. Monteiro, *"An average curvature accelerated composite gradient
  method for nonconvex smooth composite optimization problems,"* arXiv preprint
  arXiv:1909.04248, 2019 (SIAM J. Optim., 2021).

  Updates its upper curvature estimate `M_k` from the running average of
  observed local curvatures `C_i` rather than worst-case backtracking, while
  certifying exact composite stationarity `v_k = M_k (x_tilde - y_prox) +
  grad_f(y_prox) - grad_f(x_tilde) in grad_f(y_prox) + dh(y_prox)` before any
  convex-combination fallback on `y_{k+1}`.

  Attributes:
    opt_tol: Absolute stationarity tolerance `rho > 0`.
    rel_opt_tol: Optional relative stationarity tolerance vs step-0 norm.
    norm_kind: Residual norm (`'l2'` or `'linf'`).
    alpha: Average-curvature scaling factor in `(0, 1)` (default `0.5`).
    gamma: Initial curvature fraction in `(0, 1]` (`M_0 = gamma * M_bar`).
    upper_curvature: Optional smoothness bound `M_bar > 0`. Defaults to
      `oracle.curvature_upper` when available, otherwise calibrated at `x0`.
    lipschitz_init: Seed smoothness used during step-0 calibration.
    lipschitz_min: Strictly positive safeguard floor on `M_k`.
    max_iters: Maximum `AC-ACG` iterations.
  """

  opt_tol: float = 1e-5
  rel_opt_tol: float = 0.0
  norm_kind: str = "l2"
  alpha: float = 0.5
  gamma: float = 0.01
  upper_curvature: float | None = None
  lipschitz_init: float = 1.0
  lipschitz_min: float = 1e-6
  max_iters: int = 2000

  def __post_init__(self) -> None:
    """Validates `ACACG` hyperparameters."""
    if not 0.0 < self.alpha < 1.0:
      raise ValueError("`alpha` must lie strictly in (0, 1).")
    if self.gamma <= 0.0:
      raise ValueError("`gamma` must be positive.")
    if self.lipschitz_min <= 0.0:
      raise ValueError("`lipschitz_min` must be positive.")

  def solve_oracle(
      self,
      oracle: composite.CompositeOracleProtocol,
      x0: _typing.PyTree,
      *,
      opt_tol_override: _typing.Scalar | None = None,
      jit: bool = True,
      callback: _typing.CallbackFn[ACACGState] | None = None,
      time_limit: float | None = None,
  ) -> ACACGState:
    """Runs `ACACG` directly on any `CompositeOracleProtocol`."""
    x_proj = oracle.domain_project(tree.as_float_tree(x0))
    dtype = tree.infer_dtype(x_proj)

    known_L = (
        self.upper_curvature
        if self.upper_curvature is not None
        else oracle.curvature_upper
    )
    L_seed = jnp.asarray(
        known_L if known_L is not None else self.lipschitz_init,
        dtype=dtype,
    )
    ref0 = certificates.refine(oracle, x_proj, L_seed, jit=jit)
    res0 = tree.norm(ref0.v_hat, kind=self.norm_kind)
    if opt_tol_override is not None:
      target_tol = jnp.asarray(opt_tol_override, dtype=dtype)
    else:
      target_tol = jnp.asarray(self.opt_tol, dtype=dtype) + (
          jnp.asarray(self.rel_opt_tol, dtype=dtype) * res0
      )

    m_bar = (
        jnp.asarray(known_L, dtype=dtype)
        if known_L is not None
        else ref0.lipschitz_used
    )
    m_floor = jnp.maximum(
        jnp.asarray(self.lipschitz_min, dtype=dtype),
        self.gamma * m_bar,
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

    init_state = ACACGState(
        x=ref0.x_hat,
        y=ref0.x_hat,
        a_sum=jnp.asarray(0.0, dtype=dtype),
        m_est=m_floor,
        m_floor=m_floor,
        c_sum=jnp.asarray(0.0, dtype=dtype),
        good_steps=jnp.asarray(0, dtype=jnp.int32),
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

    def _cond(st: ACACGState) -> jax.Array:
      return st.status == running_code

    def _body(st: ACACGState) -> ACACGState:
      eps_sq = tree.eps(dtype) ** 2
      a_step = (1.0 + jnp.sqrt(1.0 + 4.0 * st.m_est * st.a_sum)) / (
          2.0 * st.m_est
      )
      a_next = st.a_sum + a_step
      tx = tree.convex_comb(st.a_sum, st.y, a_step, st.x)

      f_tx, grad_tx = oracle.smooth_value_and_grad(tx)
      x_next = oracle.prox(tree.add_scaled(st.x, -a_step, grad_tx), a_step)

      inv_m = 1.0 / st.m_est
      y_prox = oracle.prox(tree.add_scaled(tx, -inv_m, grad_tx), inv_m)
      f_yp, grad_yp = oracle.smooth_value_and_grad(y_prox)

      v_cand = tree.add(
          tree.scale(st.m_est, tree.sub(tx, y_prox)),
          tree.sub(grad_yp, grad_tx),
      )
      res_cand = tree.norm(v_cand, kind=self.norm_kind)
      obj_yp = f_yp + oracle.proximal_value(y_prox)

      diff_y = tree.sub(y_prox, tx)
      dist_sq = tree.norm_sq(diff_y)
      c_raw = (
          2.0
          * (f_yp - f_tx - tree.vdot(grad_tx, diff_y))
          / jnp.maximum(dist_sq, 1e-30)
      )
      c_curv = jnp.where(
          dist_sq > eps_sq,
          jnp.maximum(c_raw, jnp.asarray(0.0, dtype=dtype)),
          jnp.asarray(0.0, dtype=dtype),
      )

      bad_curv = c_curv > 0.9 * st.m_est
      y_fallback = tree.convex_comb(st.a_sum, st.y, a_step, x_next)
      y_next = jax.tree.map(
          lambda yf, yp: jnp.where(bad_curv, yf, yp),
          y_fallback,
          y_prox,
      )
      next_good = st.good_steps + jnp.where(bad_curv, 0, 1)

      c_sum_next = st.c_sum + c_curv
      denom_avg = jnp.asarray(st.outer_iters + 2, dtype=dtype)
      m_next = jnp.maximum(st.m_floor, c_sum_next / (self.alpha * denom_avg))

      cand_finite = jnp.logical_and(
          tree.all_finite(y_prox),
          jnp.logical_and(tree.all_finite(y_next), jnp.isfinite(res_cand)),
      )
      is_better = jnp.logical_and(cand_finite, res_cand <= st.best_opt_res)
      best_x = jax.tree.map(
          lambda cy, bx: jnp.where(is_better, cy, bx),
          y_prox,
          st.best_x,
      )
      best_v = jax.tree.map(
          lambda cv, bv: jnp.where(is_better, cv, bv),
          v_cand,
          st.best_subgrad,
      )
      best_res = jnp.where(is_better, res_cand, st.best_opt_res)
      best_obj = jnp.where(is_better, obj_yp, st.best_obj)

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

      return ACACGState(
          x=x_next,
          y=y_next,
          a_sum=a_next,
          m_est=m_next,
          m_floor=st.m_floor,
          c_sum=c_sum_next,
          good_steps=next_good,
          best_x=best_x,
          best_subgrad=best_v,
          best_opt_res=best_res,
          best_obj=best_obj,
          target_opt_tol=st.target_opt_tol,
          outer_iters=next_outer,
          inner_iters=st.inner_iters + 1,
          smooth_evals=st.smooth_evals + 2,
          grad_evals=st.grad_evals + 2,
          prox_evals=st.prox_evals + 2,
          status=next_status,
      )

    def _on_timeout(st: ACACGState) -> ACACGState:
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
      callback: _typing.CallbackFn[ACACGState] | None = None,
      time_limit: float | None = None,
  ) -> base.Solution:
    """Solves an unconstrained `CompositeProblem` using `ACACG`."""
    if problem.is_constrained:
      raise ValueError(
          "ACACG solves unconstrained composite problems; pass `QPAIPP`, "
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
