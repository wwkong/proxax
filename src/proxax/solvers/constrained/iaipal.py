"""Inner Accelerated Inexact Proximal AL (`IAIPAL`) solver."""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp

from proxax import _loop
from proxax import _typing
from proxax import certificates
from proxax import problem as problem_mod
from proxax import tree
from proxax.oracle import composite
from proxax.solvers import base
from proxax.solvers.inner import acg


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IAIPALState:
  """Iteration state of the `IAIPAL` solver."""

  z0: _typing.PyTree
  p0: _typing.PyTree
  z_init: _typing.PyTree
  p_init: _typing.PyTree
  penalty: jax.Array
  lam: jax.Array
  lipschitz_est: jax.Array
  nu: jax.Array
  stage: jax.Array
  local_outer_iter: jax.Array
  stage_al_base: jax.Array
  delta_psqr_avg: jax.Array
  multiplier_updates: jax.Array
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


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IAIPAL:
  """Inner Accelerated Inexact Proximal Augmented Lagrangian (`IAIPAL`) solver.

  Implements `src/frameworks/IAIPAL.m` from `wwkong/nc_opt` based on:

  - J. G. Melo and R. D. C. Monteiro, *"Iteration-complexity of an inner
    accelerated inexact proximal augmented Lagrangian method based on the
    classical Lagrangian function and a full Lagrange multiplier update,"*
    SIAM Journal on Optimization, 33(1):181--210, 2023 (`arXiv:2008.00562`);
  - W. Kong, J. G. Melo, and R. D. C. Monteiro, *"Iteration-complexity of a
    proximal augmented Lagrangian method for solving nonconvex composite
    optimization problems with nonlinear convex constraints,"* Mathematics of
    Operations Research, 48(2):1066--1094, 2023 (`arXiv:2008.07080`).

  Unlike `IALM` (which solves each AL stage subproblem to `opt_tol`), `IAIPAL`
  executes **one** inner `ACG` proximal step `(z_k, p_k) -> (z_{k+1}, p_{k+1})`
  per outer proximal AL iteration, updates multipliers according to schedule
  `k0_type in {1, 2, 3}`, and doubles the penalty `c` whenever the stagewise
  average AL progress `Delta_k` falls below the theoretical threshold.

  Attributes:
    opt_tol: Absolute Lagrangian stationarity tolerance `rho > 0`.
    feas_tol: Absolute primal feasibility & complementarity tolerance `eta > 0`.
    rel_opt_tol: Optional relative stationarity tolerance vs step-0 norm.
    rel_feas_tol: Optional relative feasibility tolerance vs step-0 norm.
    norm_kind: Residual norm (`'l2'` or `'linf'`).
    lam_init: Optional proximal step size `lambda > 0` (defaults to `1/(2m)`
      when `lower_curvature` is known, else calibrated adaptively).
    lower_curvature: Optional weak-convexity bound `m >= 0` of `f`.
    upper_curvature: Optional smoothness bound `M >= 0` of `f`.
    penalty_init: Optional initial penalty `c_0 > 0`.
    penalty_factor: Penalty growth factor (`> 1`, default `2.0`).
    penalty_max: Maximum allowed penalty parameter `c_max`.
    sigma_min: Base inner `ACG` tolerance in `(0, 1)` (default `1 / sqrt(2)`).
    sigma_type: `'constant'` or `'variable'` (`min(nu / sqrt(M_s), sigma_min)`).
    nu: Optional constant for `'variable'` `sigma` and `Delta_k` threshold.
    k0_type: Multiplier update rule (`1`: every step; `2`: every
      `ceil(c / ||A||^2)` steps; `3`: adaptive `delta_psqr_avg` trigger).
    k0_type3_alpha: Threshold constant for `k0_type == 3` (default `0.1`).
    reset_multiplier: If True, resets `p <- p_init` when `c` grows.
    reset_prox_center: If True, resets `z <- z_init` when `c` grows.
    lipschitz_init: Seed smoothness used during step-0 calibration.
    lam_min: Minimum allowed proximal step size `lambda`.
    max_outer_iters: Maximum outer proximal AL iterations.
    max_inner_iters: Maximum `ACG` iterations per proximal step.
  """

  opt_tol: float = 1e-5
  feas_tol: float = 1e-5
  rel_opt_tol: float = 0.0
  rel_feas_tol: float = 0.0
  norm_kind: str = "l2"
  lam_init: float | None = None
  lower_curvature: float | None = None
  upper_curvature: float | None = None
  penalty_init: float | None = None
  penalty_factor: float = 2.0
  penalty_max: float = 1e12
  sigma_min: float = 1.0 / math.sqrt(2.0)
  sigma_type: str = "constant"
  nu: float | None = None
  k0_type: int = 1
  k0_type3_alpha: float = 0.1
  reset_multiplier: bool = False
  reset_prox_center: bool = False
  lipschitz_init: float = 1.0
  lam_min: float = 1e-10
  max_outer_iters: int = 1000
  max_inner_iters: int = 1000

  def __post_init__(self) -> None:
    """Validates `IAIPAL` hyperparameters."""
    if not 0.0 < self.sigma_min < 1.0:
      raise ValueError("`sigma_min` must lie strictly in `(0, 1)`.")
    if self.sigma_type not in ("constant", "variable"):
      raise ValueError(
          "`sigma_type` must be 'constant' or 'variable'; "
          f"got {self.sigma_type!r}."
      )
    if self.penalty_factor <= 1.0:
      raise ValueError("`penalty_factor` must be strictly greater than 1.")
    if self.k0_type not in (1, 2, 3):
      raise ValueError(f"`k0_type` must be 1, 2, or 3; got {self.k0_type!r}.")

  def solve_state(
      self,
      problem: problem_mod.CompositeProblem,
      x0: _typing.PyTree,
      *,
      dual0: _typing.PyTree | None = None,
      jit: bool = True,
      callback: _typing.CallbackFn[IAIPALState] | None = None,
      time_limit: float | None = None,
  ) -> IAIPALState:
    """Runs `IAIPAL` and returns the full `IAIPALState`."""
    if not problem.is_constrained:
      raise ValueError(
          "IAIPAL requires a constrained `CompositeProblem`; use `AIPP`, "
          "`APD`, or `proxax.solve` for unconstrained problems."
      )

    base_oracle = problem.oracle
    z_proj = base_oracle.domain_project(tree.as_float_tree(x0))
    dtype = tree.infer_dtype(z_proj)
    bound_c = problem.bind_constraints(z_proj)
    assert bound_c is not None

    p_init = (
        bound_c.zero_dual(z_proj)
        if dual0 is None
        else tree.as_float_tree(dual0)
    )

    known_m = (
        self.lower_curvature
        if self.lower_curvature is not None
        else base_oracle.curvature_lower
    )
    known_M = (
        self.upper_curvature
        if self.upper_curvature is not None
        else base_oracle.curvature_upper
    )

    L_seed = jnp.asarray(
        known_M if known_M is not None else self.lipschitz_init,
        dtype=dtype,
    )
    ref_base = certificates.refine(base_oracle, z_proj, L_seed, jit=jit)
    m_upper_base = (
        jnp.asarray(known_M, dtype=dtype)
        if known_M is not None
        else ref_base.lipschitz_used
    )
    m_upper_base = jnp.maximum(m_upper_base, jnp.asarray(1e-4, dtype=dtype))

    adaptive_lam = self.lam_init is None and known_m is None
    if self.lam_init is not None:
      lam0 = jnp.asarray(self.lam_init, dtype=dtype)
    elif known_m is not None:
      m_arr = jnp.asarray(known_m, dtype=dtype)
      lam0 = jnp.where(
          m_arr > 0.0,
          0.5 / jnp.maximum(m_arr, jnp.asarray(1e-12, dtype=dtype)),
          2.0 / m_upper_base,
      )
    else:
      lam0 = jnp.maximum(
          1.0 / m_upper_base,
          jnp.asarray(self.lam_min, dtype=dtype),
      )

    op_norm_sq = (
        jnp.asarray(bound_c.operator_norm_sq, dtype=dtype)
        if bound_c.operator_norm_sq is not None
        else jnp.asarray(1.0, dtype=dtype)
    )
    op_norm_sq = jnp.maximum(op_norm_sq, jnp.asarray(1e-6, dtype=dtype))

    if self.penalty_init is not None:
      c0 = jnp.asarray(self.penalty_init, dtype=dtype)
    else:
      c0 = jnp.maximum(jnp.asarray(1.0, dtype=dtype), m_upper_base / op_norm_sq)

    nu_arr = (
        jnp.asarray(self.nu, dtype=dtype)
        if self.nu is not None
        else jnp.sqrt(self.sigma_min * (lam0 * m_upper_base + 1.0))
    )

    pen0 = composite.PenalizedOracle(
        base=base_oracle,
        constraint=bound_c,
        multiplier=p_init,
        penalty=c0,
        dampen=0.0,
    )
    ref0 = certificates.refine(pen0, z_proj, m_upper_base, jit=jit)
    p0_hat = pen0.effective_dual(ref0.x_hat)
    al_feas0 = tree.norm(
        tree.scale(1.0 / c0, tree.sub(p0_hat, p_init)),
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

    init_state = IAIPALState(
        z0=z_proj,
        p0=p_init,
        z_init=z_proj,
        p_init=p_init,
        penalty=c0,
        lam=lam0,
        lipschitz_est=ref0.lipschitz_used,
        nu=nu_arr,
        stage=jnp.asarray(1, dtype=jnp.int32),
        local_outer_iter=jnp.asarray(1, dtype=jnp.int32),
        stage_al_base=jnp.asarray(0.0, dtype=dtype),
        delta_psqr_avg=jnp.asarray(0.0, dtype=dtype),
        multiplier_updates=jnp.asarray(0, dtype=jnp.int32),
        best_x=ref0.x_hat,
        best_dual=p0_hat,
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

    def _cond(st: IAIPALState) -> jax.Array:
      return st.status == running_code

    def _body(st: IAIPALState) -> IAIPALState:
      pen_oracle = composite.PenalizedOracle(
          base=base_oracle,
          constraint=bound_c,
          multiplier=st.p0,
          penalty=st.penalty,
          dampen=0.0,
      )

      inv_lam = 1.0 / st.lam
      if known_m is not None and not adaptive_lam:
        m_arr = jnp.asarray(known_m, dtype=dtype)
        smooth_reg = jnp.minimum(m_arr, 0.5 * inv_lam)
        prox_reg = jnp.maximum(
            jnp.asarray(1e-6, dtype=dtype),
            inv_lam - smooth_reg,
        )
      else:
        smooth_reg = 0.5 * inv_lam
        prox_reg = 0.5 * inv_lam

      sub_oracle = composite.ProxRegularizedOracle(
          base=pen_oracle,
          center=st.z0,
          smooth_reg=smooth_reg,
          prox_reg=prox_reg,
      )

      m_s = st.lam * st.lipschitz_est + 1.0
      if self.sigma_type == "variable":
        sigma_k = jnp.minimum(
            st.nu / jnp.sqrt(jnp.maximum(m_s, 1.0)),
            jnp.asarray(self.sigma_min, dtype=dtype),
        )
      else:
        sigma_k = jnp.asarray(self.sigma_min, dtype=dtype)

      acg_cfg = acg.ACGConfig(
          sigma=sigma_k,
          max_iters=self.max_inner_iters,
      )
      acg_st = acg.run_acg(
          sub_oracle,
          st.z0,
          prox_center=st.z0,
          lam=st.lam,
          lipschitz_init=st.lipschitz_est + smooth_reg,
          config=acg_cfg,
          check_convexity=adaptive_lam,
          jit=jit,
      )

      noncvx_code = jnp.asarray(
          int(acg.ACGStatus.NONCONVEX_DETECTED), dtype=jnp.int32
      )
      noncvx_hit = jnp.logical_and(
          jnp.asarray(adaptive_lam),
          acg_st.status == noncvx_code,
      )

      est_al_L = jnp.maximum(
          acg_st.lipschitz - smooth_reg,
          0.5 * st.lipschitz_est,
      )
      ref = certificates.refine(pen_oracle, acg_st.y, est_al_L, jit=jit)
      p_hat = pen_oracle.effective_dual(ref.x_hat)
      w_hat = ref.v_hat
      opt_res = tree.norm(w_hat, kind=self.norm_kind)
      al_feas = tree.norm(
          tree.scale(1.0 / st.penalty, tree.sub(p_hat, st.p0)),
          kind=self.norm_kind,
      )
      raw_feas = tree.norm(bound_c.residual(ref.x_hat), kind=self.norm_kind)
      feas_res = jnp.maximum(al_feas, raw_feas)
      cand_obj = base_oracle.value(ref.x_hat)
      cand_score = jnp.maximum(
          opt_res / jnp.maximum(st.target_opt_tol, tiny),
          feas_res / jnp.maximum(st.target_feas_tol, tiny),
      )

      cand_finite = jnp.logical_and(
          tree.all_finite(ref.x_hat),
          jnp.logical_and(jnp.isfinite(opt_res), jnp.isfinite(feas_res)),
      )
      is_better = jnp.logical_and(cand_finite, cand_score <= st.best_kkt_score)
      best_x = jax.tree.map(
          lambda cx, bx: jnp.where(is_better, cx, bx),
          ref.x_hat,
          st.best_x,
      )
      best_p = jax.tree.map(
          lambda cp, bp: jnp.where(is_better, cp, bp),
          p_hat,
          st.best_dual,
      )
      best_v = jax.tree.map(
          lambda cv, bv: jnp.where(is_better, cv, bv),
          w_hat,
          st.best_subgrad,
      )
      best_opt = jnp.where(is_better, opt_res, st.best_opt_res)
      best_feas = jnp.where(is_better, feas_res, st.best_feas_res)
      best_score = jnp.where(is_better, cand_score, st.best_kkt_score)
      best_obj = jnp.where(is_better, cand_obj, st.best_obj)

      x_acg = acg_st.y
      p_cand = pen_oracle.effective_dual(x_acg)
      k0_period = jnp.maximum(
          jnp.asarray(1, dtype=jnp.int32),
          jnp.ceil(st.penalty / op_norm_sq).astype(jnp.int32),
      )
      if self.k0_type == 1:
        do_mult = jnp.asarray(True)
      elif self.k0_type == 2:
        do_mult = jnp.logical_or(
            st.local_outer_iter == 1,
            jnp.mod(st.local_outer_iter - 1, k0_period) == 0,
        )
      else:
        do_mult = jnp.logical_or(
            st.local_outer_iter == 1,
            st.delta_psqr_avg / st.lam <= self.k0_type3_alpha / st.penalty,
        )

      p_updated = jax.tree.map(
          lambda pc, p_old: jnp.where(do_mult, pc, p_old),
          p_cand,
          st.p0,
      )
      dp_sq = tree.norm_sq(tree.sub(p_updated, st.p0))
      loc_f = st.local_outer_iter.astype(dtype)
      next_dpsqr = (st.delta_psqr_avg * (loc_f - 1.0) + dp_sq) / jnp.maximum(
          loc_f, 1.0
      )

      delta_oracle = composite.PenalizedOracle(
          base=base_oracle,
          constraint=bound_c,
          multiplier=p_updated,
          penalty=st.penalty,
          dampen=0.0,
      )
      stage_al_val = delta_oracle.value(x_acg)
      is_first_in_stage = st.local_outer_iter == 1
      denom_cyc = jnp.maximum(loc_f - 1.0, 1.0)
      delta_k = (
          st.stage_al_base
          - stage_al_val
          - 0.5 * tree.norm_sq(p_updated) / st.penalty
      ) / denom_cyc
      delta_mult = (
          st.lam
          * (1.0 - sigma_k**2)
          / (2.0 * (1.0 + 2.0 * st.nu) ** 2)
      )
      grow_c = jnp.logical_and(
          jnp.logical_not(is_first_in_stage),
          jnp.logical_and(
              jnp.logical_not(noncvx_hit),
              delta_k <= (st.target_opt_tol**2) * delta_mult,
          ),
      )

      c_grown = jnp.minimum(
          st.penalty * self.penalty_factor,
          jnp.asarray(self.penalty_max, dtype=dtype),
      )
      c_next = jnp.where(grow_c, c_grown, st.penalty)
      lip_next = ref.lipschitz_used * (c_next / st.penalty)
      stage_next = jnp.where(grow_c, st.stage + 1, st.stage)
      local_next = jnp.where(
          noncvx_hit,
          st.local_outer_iter,
          jnp.where(
              grow_c,
              jnp.asarray(1, dtype=jnp.int32),
              st.local_outer_iter + 1,
          ),
      )
      base_next = jnp.where(
          jnp.logical_and(is_first_in_stage, jnp.logical_not(noncvx_hit)),
          stage_al_val,
          st.stage_al_base,
      )
      dpsqr_next = jnp.where(
          grow_c,
          jnp.asarray(0.0, dtype=dtype),
          jnp.where(noncvx_hit, st.delta_psqr_avg, next_dpsqr),
      )

      p_after_grow = (
          st.p_init if self.reset_multiplier else p_updated
      )
      z_after_grow = (
          st.z_init if self.reset_prox_center else x_acg
      )
      p_acc = jax.tree.map(
          lambda pg, pu: jnp.where(grow_c, pg, pu),
          p_after_grow,
          p_updated,
      )
      z_acc = jax.tree.map(
          lambda zg, zu: jnp.where(grow_c, zg, zu),
          z_after_grow,
          x_acg,
      )

      p_next = jax.tree.map(
          lambda p_old, pa: jnp.where(noncvx_hit, p_old, pa),
          st.p0,
          p_acc,
      )
      z_next = jax.tree.map(
          lambda z_old, za: jnp.where(noncvx_hit, z_old, za),
          st.z0,
          z_acc,
      )
      lam_next = jnp.where(
          noncvx_hit,
          jnp.maximum(0.5 * st.lam, jnp.asarray(self.lam_min, dtype=dtype)),
          st.lam,
      )
      mult_upd_next = st.multiplier_updates + jnp.where(
          jnp.logical_and(do_mult, jnp.logical_not(noncvx_hit)),
          1,
          0,
      )

      next_outer = st.outer_iters + 1
      kkt_converged = jnp.logical_and(
          best_opt <= st.target_opt_tol,
          best_feas <= st.target_feas_tol,
      )
      ls_fail_code = jnp.asarray(
          int(acg.ACGStatus.LINE_SEARCH_FAILED), dtype=jnp.int32
      )
      next_status = jnp.where(
          jnp.logical_not(cand_finite),
          jnp.asarray(int(base.SolveStatus.NON_FINITE), dtype=jnp.int32),
          jnp.where(
              acg_st.status == ls_fail_code,
              jnp.asarray(
                  int(base.SolveStatus.LINE_SEARCH_FAILED), dtype=jnp.int32
              ),
              jnp.where(
                  kkt_converged,
                  jnp.asarray(int(base.SolveStatus.CONVERGED), dtype=jnp.int32),
                  jnp.where(
                      next_outer >= self.max_outer_iters,
                      jnp.asarray(
                          int(base.SolveStatus.MAX_ITER_REACHED),
                          dtype=jnp.int32,
                      ),
                      running_code,
                  ),
              ),
          ),
      )

      return IAIPALState(
          z0=z_next,
          p0=p_next,
          z_init=st.z_init,
          p_init=st.p_init,
          penalty=c_next,
          lam=lam_next,
          lipschitz_est=lip_next,
          nu=st.nu,
          stage=stage_next,
          local_outer_iter=local_next,
          stage_al_base=base_next,
          delta_psqr_avg=dpsqr_next,
          multiplier_updates=mult_upd_next,
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
          inner_iters=st.inner_iters + acg_st.iteration,
          smooth_evals=(
              st.smooth_evals + acg_st.smooth_evals + ref.smooth_evals + 2
          ),
          grad_evals=st.grad_evals + acg_st.grad_evals + ref.grad_evals,
          prox_evals=st.prox_evals + acg_st.prox_evals + ref.prox_evals,
          status=next_status,
      )

    def _on_timeout(st: IAIPALState) -> IAIPALState:
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
      callback: _typing.CallbackFn[IAIPALState] | None = None,
      time_limit: float | None = None,
  ) -> base.Solution:
    """Solves a constrained `CompositeProblem` using `IAIPAL`."""
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
