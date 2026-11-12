"""Smoothed Proximal Augmented Lagrangian (`s-Prox-ALM`, Zhang & Luo 2020)."""

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
class SProxALMState:
  """Iteration state of the single-loop `SProxALM` (`s-Prox-ALM`) solver."""

  x: _typing.PyTree
  y: _typing.PyTree
  z: _typing.PyTree
  step_c: jax.Array
  prox_p: jax.Array
  best_x: _typing.PyTree
  best_dual: _typing.PyTree
  best_subgrad: _typing.PyTree
  best_opt_res: jax.Array
  best_feas_res: jax.Array
  best_kkt_score: jax.Array
  best_obj: jax.Array
  target_opt_tol: jax.Array
  target_feas_tol: jax.Array
  outer_iters: jax.Array
  inner_iters: jax.Array
  smooth_evals: jax.Array
  grad_evals: jax.Array
  prox_evals: jax.Array
  status: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _SProxBacktrackCarry:
  """Backtracking carry calibrating primal step size `c <= 1 / (2 L_K)`."""

  step_c: jax.Array
  x_trial: _typing.PyTree
  f_pen_trial: jax.Array
  accepted: jax.Array
  trials: jax.Array


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class SProxALM:
  """Smoothed Proximal Augmented Lagrangian (`s-Prox-ALM`) solver.

  Implements `src/frameworks/sProxALM.m` from `wwkong/nc_opt` based on J. Zhang
  and Z.-Q. Luo, *"A global dual error bound and its application to the analysis
  of linearly constrained nonconvex optimization,"* SIAM Journal on
  Optimization, 32(3):2319--2346, 2022 (`arXiv:2006.16440`).

  Generalizes `sProxALM.m` from equality constraints to arbitrary `ConvexSet`
  PyTree constraints `A(x) in C` via `PenalizedOracle`, fixes the proximal
  step-size scaling (`prox_{c h}`), automatically calibrates `c <= 1 / (2 L_K)`
  via backtracking when `(L, ||A||_op)` are unknown, and certifies exact KKT
  stationarity `v_k in grad_f(x_k) + dh(x_k) + A^*(p_hat_k)`.

  Attributes:
    opt_tol: Absolute Lagrangian stationarity tolerance `rho > 0`.
    feas_tol: Absolute primal feasibility & complementarity tolerance `eta > 0`.
    rel_opt_tol: Optional relative stationarity tolerance vs step-0 norm.
    rel_feas_tol: Optional relative feasibility tolerance vs step-0 norm.
    norm_kind: Residual norm (`'l2'` or `'linf'`).
    penalty: Augmented Lagrangian penalty `Gamma > 0` (default `10.0`).
    dual_step: Optional dual ascent step size `alpha > 0` (defaults to
      `0.25 * penalty`).
    prox_reg: Optional smoothing proximal regularization `p > 0` on
      `0.5 * p * ||x - z||_2^2`.
    primal_step: Optional fixed primal step size `c > 0`. When `None`,
      calibrated automatically to satisfy `c <= 1 / (2 L_K)`.
    beta: Smoothing center rate in `(0, 1]` (`z <- z + beta * (x - z)`).
    lipschitz_init: Seed smoothness used during step-0 calibration.
    max_iters: Maximum single-loop `s-Prox-ALM` iterations.
    max_backtracks: Maximum primal step-size halvings per iteration.
  """

  opt_tol: float = 1e-5
  feas_tol: float = 1e-5
  rel_opt_tol: float = 0.0
  rel_feas_tol: float = 0.0
  norm_kind: str = "l2"
  penalty: float = 10.0
  dual_step: float | None = None
  prox_reg: float | None = None
  primal_step: float | None = None
  beta: float = 0.5
  lipschitz_init: float = 1.0
  max_iters: int = 4000
  max_backtracks: int = 30

  def __post_init__(self) -> None:
    """Validates `SProxALM` hyperparameters."""
    if self.penalty <= 0.0:
      raise ValueError("`penalty` (`Gamma`) must be positive.")
    if self.dual_step is not None and self.dual_step <= 0.0:
      raise ValueError("`dual_step` (`alpha`) must be positive.")
    if self.prox_reg is not None and self.prox_reg <= 0.0:
      raise ValueError("`prox_reg` (`p`) must be positive.")
    if self.primal_step is not None and self.primal_step <= 0.0:
      raise ValueError("`primal_step` (`c`) must be positive.")
    if not 0.0 < self.beta <= 1.0:
      raise ValueError("`beta` must lie in `(0, 1]`.")

  def solve_state(
      self,
      problem: problem_mod.CompositeProblem,
      x0: _typing.PyTree,
      *,
      dual0: _typing.PyTree | None = None,
      jit: bool = True,
      callback: _typing.CallbackFn[SProxALMState] | None = None,
      time_limit: float | None = None,
  ) -> SProxALMState:
    """Runs `SProxALM` and returns the full `SProxALMState`."""
    if not problem.is_constrained:
      raise ValueError(
          "SProxALM requires a constrained `CompositeProblem`; use `AIPP`, "
          "`APD`, or `proxax.solve` for unconstrained problems."
      )

    base_oracle = problem.oracle
    x_proj = base_oracle.domain_project(tree.as_float_tree(x0))
    dtype = tree.infer_dtype(x_proj)
    bound_c = problem.bind_constraints(x_proj)
    assert bound_c is not None

    y_init = (
        bound_c.zero_dual(x_proj)
        if dual0 is None
        else tree.as_float_tree(dual0)
    )
    gamma_arr = jnp.asarray(self.penalty, dtype=dtype)
    alpha_val = (
        self.dual_step if self.dual_step is not None else 0.25 * self.penalty
    )
    alpha_arr = jnp.asarray(alpha_val, dtype=dtype)

    L_seed = jnp.asarray(self.lipschitz_init, dtype=dtype)
    ref_base = certificates.refine(base_oracle, x_proj, L_seed, jit=jit)

    if self.prox_reg is not None:
      p_arr = jnp.asarray(self.prox_reg, dtype=dtype)
    else:
      m_known = (
          jnp.asarray(base_oracle.curvature_lower, dtype=dtype)
          if base_oracle.curvature_lower is not None
          else jnp.asarray(0.0, dtype=dtype)
      )
      p_arr = jnp.maximum(
          jnp.maximum(2.0 * m_known, 2.0 * ref_base.lipschitz_used),
          jnp.asarray(1.0, dtype=dtype),
      )

    pen0 = composite.PenalizedOracle(
        base=base_oracle,
        constraint=bound_c,
        multiplier=y_init,
        penalty=gamma_arr,
        dampen=0.0,
    )
    ref0 = certificates.refine(pen0, x_proj, ref_base.lipschitz_used, jit=jit)
    p0 = pen0.effective_dual(ref0.x_hat)
    al_feas0 = tree.norm(
        tree.scale(1.0 / gamma_arr, tree.sub(p0, y_init)),
        kind=self.norm_kind,
    )
    raw_feas0 = tree.norm(bound_c.residual(ref0.x_hat), kind=self.norm_kind)
    feas_res0 = jnp.maximum(al_feas0, raw_feas0)
    opt_res0 = tree.norm(ref0.v_hat, kind=self.norm_kind)
    obj0 = base_oracle.value(ref0.x_hat)

    target_opt = jnp.asarray(self.opt_tol, dtype=dtype) + (
        jnp.asarray(self.rel_opt_tol, dtype=dtype) * opt_res0
    )
    target_feas = jnp.asarray(self.feas_tol, dtype=dtype) + (
        jnp.asarray(self.rel_feas_tol, dtype=dtype) * feas_res0
    )

    use_fixed_c = self.primal_step is not None or (
        base_oracle.curvature_upper is not None
        and bound_c.operator_norm_sq is not None
    )
    if self.primal_step is not None:
      c_init = jnp.asarray(self.primal_step, dtype=dtype)
    elif (
        base_oracle.curvature_upper is not None
        and bound_c.operator_norm_sq is not None
    ):
      l_tot = (
          jnp.asarray(base_oracle.curvature_upper, dtype=dtype)
          + p_arr
          + gamma_arr * jnp.asarray(bound_c.operator_norm_sq, dtype=dtype)
      )
      c_init = 1.0 / jnp.maximum(2.0 * l_tot, jnp.asarray(1e-6, dtype=dtype))
    else:
      c_init = 1.0 / jnp.maximum(
          2.0 * (ref0.lipschitz_used + p_arr),
          jnp.asarray(1e-6, dtype=dtype),
      )

    tiny = jnp.asarray(1e-30, dtype=dtype)
    score0 = jnp.maximum(
        opt_res0 / jnp.maximum(target_opt, tiny),
        feas_res0 / jnp.maximum(target_feas, tiny),
    )
    finite0 = jnp.logical_and(
        tree.all_finite(ref0.x_hat),
        jnp.logical_and(jnp.isfinite(opt_res0), jnp.isfinite(feas_res0)),
    )
    converged0 = jnp.logical_and(
        opt_res0 <= target_opt,
        feas_res0 <= target_feas,
    )
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

    init_state = SProxALMState(
        x=x_proj,
        y=y_init,
        z=x_proj,
        step_c=c_init,
        prox_p=p_arr,
        best_x=ref0.x_hat,
        best_dual=p0,
        best_subgrad=ref0.v_hat,
        best_opt_res=opt_res0,
        best_feas_res=feas_res0,
        best_kkt_score=score0,
        best_obj=obj0,
        target_opt_tol=target_opt,
        target_feas_tol=target_feas,
        outer_iters=jnp.asarray(0, dtype=jnp.int32),
        inner_iters=jnp.asarray(0, dtype=jnp.int32),
        smooth_evals=ref_base.smooth_evals + ref0.smooth_evals + 1,
        grad_evals=ref_base.grad_evals + ref0.grad_evals,
        prox_evals=ref_base.prox_evals + ref0.prox_evals,
        status=status0,
    )

    def _cond(st: SProxALMState) -> jax.Array:
      return st.status == running_code

    def _body(st: SProxALMState) -> SProxALMState:
      eps_val = tree.eps(dtype)
      _, y_next = bound_c.penalty_and_dual(
          st.x,
          multiplier=st.y,
          penalty=alpha_arr,
          dampen=0.0,
      )
      pen_oracle = composite.PenalizedOracle(
          base=base_oracle,
          constraint=bound_c,
          multiplier=y_next,
          penalty=gamma_arr,
          dampen=0.0,
      )

      f_pen0, g_pen0 = pen_oracle.smooth_value_and_grad(st.x)
      diff_x0_z0 = tree.sub(st.x, st.z)
      k_val0 = f_pen0 + 0.5 * st.prox_p * tree.norm_sq(diff_x0_z0)
      grad_k0 = tree.add_scaled(g_pen0, st.prox_p, diff_x0_z0)
      slack = 10.0 * eps_val * (1.0 + jnp.abs(k_val0))

      if use_fixed_c:
        c_used = st.step_c
        x_next = base_oracle.prox(
            tree.add_scaled(st.x, -c_used, grad_k0),
            c_used,
        )
        bt_trials = jnp.asarray(1, dtype=jnp.int32)
        extra_s = jnp.asarray(0, dtype=jnp.int32)
      else:

        def _trial_c(
            c_val: jax.Array,
        ) -> tuple[_typing.PyTree, jax.Array, jax.Array]:
          xt = base_oracle.prox(tree.add_scaled(st.x, -c_val, grad_k0), c_val)
          f_pt = pen_oracle.smooth_value(xt)
          k_t = f_pt + 0.5 * st.prox_p * tree.norm_sq(tree.sub(xt, st.z))
          dx = tree.sub(xt, st.x)
          quad = (
              k_val0
              + tree.vdot(grad_k0, dx)
              + (0.25 / jnp.maximum(c_val, tiny)) * tree.norm_sq(dx)
          )
          return xt, f_pt, k_t <= quad + slack

        xt0, fp0, ok0 = _trial_c(st.step_c)
        bt_init = _SProxBacktrackCarry(
            step_c=st.step_c,
            x_trial=xt0,
            f_pen_trial=fp0,
            accepted=ok0,
            trials=jnp.asarray(1, dtype=jnp.int32),
        )

        def _bt_cond(bc: _SProxBacktrackCarry) -> jax.Array:
          return jnp.logical_and(
              jnp.logical_not(bc.accepted),
              bc.trials < self.max_backtracks,
          )

        def _bt_body(bc: _SProxBacktrackCarry) -> _SProxBacktrackCarry:
          next_c = 0.5 * bc.step_c
          xt, fp, ok = _trial_c(next_c)
          return _SProxBacktrackCarry(
              step_c=next_c,
              x_trial=xt,
              f_pen_trial=fp,
              accepted=ok,
              trials=bc.trials + 1,
          )

        bt_fin = _loop.while_loop(_bt_cond, _bt_body, bt_init, jit=jit)
        c_used = bt_fin.step_c
        x_next = bt_fin.x_trial
        bt_trials = bt_fin.trials
        extra_s = bt_fin.trials

      z_next = jax.tree.map(
          lambda z0_i, xn_i: z0_i + self.beta * (xn_i - z0_i),
          st.z,
          x_next,
      )

      _, g_pen_next = pen_oracle.smooth_value_and_grad(x_next)
      p_hat = pen_oracle.effective_dual(x_next)
      v_cand = tree.add(
          tree.sub(g_pen_next, grad_k0),
          tree.scale(1.0 / jnp.maximum(c_used, tiny), tree.sub(st.x, x_next)),
      )

      cand_opt = tree.norm(v_cand, kind=self.norm_kind)
      al_feas = tree.norm(
          tree.scale(1.0 / gamma_arr, tree.sub(p_hat, y_next)),
          kind=self.norm_kind,
      )
      raw_feas = tree.norm(bound_c.residual(x_next), kind=self.norm_kind)
      cand_feas = jnp.maximum(al_feas, raw_feas)
      cand_obj = base_oracle.value(x_next)
      cand_score = jnp.maximum(
          cand_opt / jnp.maximum(st.target_opt_tol, tiny),
          cand_feas / jnp.maximum(st.target_feas_tol, tiny),
      )

      cand_finite = jnp.logical_and(
          tree.all_finite(x_next),
          jnp.logical_and(jnp.isfinite(cand_opt), jnp.isfinite(cand_feas)),
      )
      is_better = jnp.logical_and(cand_finite, cand_score <= st.best_kkt_score)
      best_x = jax.tree.map(
          lambda cx, bx: jnp.where(is_better, cx, bx),
          x_next,
          st.best_x,
      )
      best_p = jax.tree.map(
          lambda cp, bp: jnp.where(is_better, cp, bp),
          p_hat,
          st.best_dual,
      )
      best_v = jax.tree.map(
          lambda cv, bv: jnp.where(is_better, cv, bv),
          v_cand,
          st.best_subgrad,
      )
      best_opt = jnp.where(is_better, cand_opt, st.best_opt_res)
      best_feas = jnp.where(is_better, cand_feas, st.best_feas_res)
      best_score = jnp.where(is_better, cand_score, st.best_kkt_score)
      best_obj = jnp.where(is_better, cand_obj, st.best_obj)

      next_outer = st.outer_iters + 1
      kkt_converged = jnp.logical_and(
          best_opt <= st.target_opt_tol,
          best_feas <= st.target_feas_tol,
      )
      next_status = jnp.where(
          jnp.logical_not(cand_finite),
          jnp.asarray(int(base.SolveStatus.NON_FINITE), dtype=jnp.int32),
          jnp.where(
              kkt_converged,
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

      return SProxALMState(
          x=x_next,
          y=y_next,
          z=z_next,
          step_c=c_used,
          prox_p=st.prox_p,
          best_x=best_x,
          best_dual=best_p,
          best_subgrad=best_v,
          best_opt_res=best_opt,
          best_feas_res=best_feas,
          best_kkt_score=best_score,
          best_obj=best_obj,
          target_opt_tol=st.target_opt_tol,
          target_feas_tol=st.target_feas_tol,
          outer_iters=next_outer,
          inner_iters=st.inner_iters + bt_trials,
          smooth_evals=st.smooth_evals + 3 + extra_s,
          grad_evals=st.grad_evals + 2,
          prox_evals=st.prox_evals + bt_trials,
          status=next_status,
      )

    def _on_timeout(st: SProxALMState) -> SProxALMState:
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
      callback: _typing.CallbackFn[SProxALMState] | None = None,
      time_limit: float | None = None,
  ) -> base.Solution:
    """Solves a constrained `CompositeProblem` using `SProxALM`."""
    fin = self.solve_state(
        problem,
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
        feas_residual=fin.best_feas_res,
        obj_val=fin.best_obj,
    )
    return base.Solution(
        x=fin.best_x,
        dual=fin.best_dual,
        subgrad=fin.best_subgrad,
        status=fin.status,
        stats=stats,
    )
