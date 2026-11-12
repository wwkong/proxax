"""Dampened Inexact Proximal Augmented Lagrangian (`iALM`) solver."""

from __future__ import annotations

import dataclasses
import time

import jax
import jax.numpy as jnp

from proxax import _loop
from proxax import _typing
from proxax import certificates
from proxax import problem as problem_mod
from proxax import tree
from proxax.oracle import composite
from proxax.solvers import base
from proxax.solvers.unconstrained import aipp as aipp_mod
from proxax.solvers.unconstrained import apd as apd_mod


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IALMState:
  """Outer Augmented Lagrangian iteration state of the `IALM` solver."""

  primal: _typing.PyTree
  multiplier: _typing.PyTree
  best_x: _typing.PyTree
  best_dual: _typing.PyTree
  best_subgrad: _typing.PyTree
  best_opt_res: jax.Array
  best_feas_res: jax.Array
  best_kkt_score: jax.Array
  best_obj: jax.Array
  prev_feas_res: jax.Array
  target_opt_tol: jax.Array
  target_feas_tol: jax.Array
  penalty: jax.Array
  lam: jax.Array
  lower_curvature: jax.Array
  upper_curvature: jax.Array
  outer_iters: jax.Array
  inner_iters: jax.Array
  smooth_evals: jax.Array
  grad_evals: jax.Array
  prox_evals: jax.Array
  status: jax.Array


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IALM:
  """Dampened Inexact Proximal Augmented Lagrangian (`iALM`) solver.

  Unifies three closely related proximal Augmented Lagrangian methods from
  `wwkong/nc_opt` on top of `PenalizedOracle`:

  - **Parameter-Free `iALM` (`subproblem_solver='apd'`, Section 7 of Kong
    2024):** Solves each AL subproblem via parameter-free `APD` (`PF.APD` +
    `PF.ACG`) without requiring weak-convexity or smoothness bounds.
  - **`NL-IAPIAL` (`subproblem_solver='aipp'`, `dampen=0.0`):** Undampened
    proximal AL with `AIPP` subproblem solves and conditional penalty growth.
  - **`AIDAL` (`dampen` in `(0, 1)`, `dual_step` in `(0, 1]`):** Dampened
    Moreau-envelope AL update `p_bar_k = (1 - dampen) * p_k` with fractional
    dual step size `dual_step * c_k` (Kong, Melo, and Monteiro 2023).

  Returns a certified primal-dual-subgradient triple `(x_hat, p_hat, v_hat)`
  satisfying:

  - `v_hat in grad_f(x_hat) + dh(x_hat) + A^*(p_hat)` with
    `||v_hat|| <= opt_tol`,
  - `p_hat in N_C(y_tilde)` for `y_tilde = Pi_C(A(x_hat) + p_bar / c_k) in C`,
  - `||A(x_hat) - y_tilde|| <= feas_tol` and
    `||A(x_hat) - Pi_C(A(x_hat))|| <= feas_tol`.

  Attributes:
    opt_tol: Absolute Lagrangian stationarity tolerance `rho > 0`.
    feas_tol: Absolute primal feasibility & complementarity tolerance `eta > 0`.
    rel_opt_tol: Optional relative stationarity tolerance vs step-0 norm.
    rel_feas_tol: Optional relative feasibility tolerance vs step-0 norm.
    norm_kind: Residual norm (`'l2'` or `'linf'`).
    subproblem_solver: Inner solver (`'apd'` or `'aipp'`).
    dampen: Multiplier dampening factor `theta` in `[0, 1)` (`0.0` = undampened
      `NL-IAPIAL`; `(0, 1)` = dampened `AIDAL`).
    dual_step: Dual update step fraction `chi` in `(0, 1]`.
    dual_bound: Optional L2 norm cap on dual multipliers `p_{k+1}`.
    penalty_init: Initial quadratic penalty parameter `c_0 > 0`.
    penalty_factor: Multiplicative penalty growth factor (`> 1`).
    penalty_max: Maximum allowed penalty parameter `c_max`.
    feas_decay: Required feasibility contraction factor in `[0, 1)` to keep
      `c_k` unchanged; otherwise `c_{k+1} = min(penalty_factor * c_k, c_max)`.
    max_alm_iters: Maximum number of outer Augmented Lagrangian stages.
    alpha: `APD` multiplicative growth factor (`> 1`) for `m_hat`.
    beta: `APD` multiplicative growth factor (`> 1`) for inner smoothness `L`.
    theta: `APD` proximal descent constant (`> 2`, default `4.0`).
    rho: Optional `APD` inner relative proximal error tolerance in `(0, 1)`.
    decay_curvature: Whether `APD` applies practical curvature decay Eq. (6.1).
    lower_curvature_init: Optional initial lower curvature `m_0 > 0` for `APD`.
    lower_curvature_min: Floor `m_min > 0` guarding `APD` curvature decay.
    lower_curvature_max: Maximum allowed `APD` lower curvature estimate.
    sigma: `AIPP` inner `ACG` relative inexactness parameter in `(0, 1)`.
    lam_init: Optional initial proximal step size `lambda_0 > 0` for `AIPP`.
    lam_min: Minimum allowed `AIPP` proximal step size.
    lam_max: Maximum allowed `AIPP` proximal step size.
    adaptive_prox: Whether `AIPP` adapts `lam` dynamically (`None` = auto).
    lipschitz_init: Initial smoothness seed `L_0 > 0` for `f`.
    max_subproblem_iters: Maximum outer proximal steps per AL stage.
    max_inner_iters: Maximum inner `PF.ACG` / `ACG` steps per proximal step.
  """

  opt_tol: float = 1e-5
  feas_tol: float = 1e-5
  rel_opt_tol: float = 0.0
  rel_feas_tol: float = 0.0
  norm_kind: str = "l2"
  subproblem_solver: str = "apd"
  dampen: float = 0.0
  dual_step: float = 1.0
  dual_bound: float | None = None
  penalty_init: float = 1.0
  penalty_factor: float = 2.0
  penalty_max: float = 1e12
  feas_decay: float = 0.5
  max_alm_iters: int = 50
  alpha: float = 2.0
  beta: float = 2.0
  theta: float = 4.0
  rho: float | None = None
  decay_curvature: bool = True
  lower_curvature_init: float | None = None
  lower_curvature_min: float = 1e-6
  lower_curvature_max: float = 1e12
  sigma: float = 0.3
  lam_init: float | None = None
  lam_min: float = 1e-10
  lam_max: float = 1e8
  adaptive_prox: bool | None = None
  lipschitz_init: float = 1.0
  max_subproblem_iters: int = 250
  max_inner_iters: int = 1000

  def __post_init__(self) -> None:
    """Validates `IALM` hyperparameters."""
    key = self.subproblem_solver.strip().lower().replace("-", "_")
    if key not in ("apd", "pf_apd", "aipp"):
      raise ValueError(
          f"Unsupported `subproblem_solver` {self.subproblem_solver!r}; "
          "expected 'apd' or 'aipp'."
      )
    if not (0.0 <= self.dampen < 1.0):
      raise ValueError("`dampen` must lie in `[0, 1)`.")
    if not (0.0 < self.dual_step <= 1.0):
      raise ValueError("`dual_step` must lie in `(0, 1]`.")
    if self.penalty_factor <= 1.0:
      raise ValueError("`penalty_factor` must be strictly greater than 1.")
    if not (0.0 <= self.feas_decay < 1.0):
      raise ValueError("`feas_decay` must lie in `[0, 1)`.")
    if self.dual_bound is not None and self.dual_bound < 0.0:
      raise ValueError("`dual_bound` must be non-negative.")

  @property
  def _uses_apd(self) -> bool:
    key = self.subproblem_solver.strip().lower().replace("-", "_")
    return key in ("apd", "pf_apd")

  def solve_state(
      self,
      problem: problem_mod.CompositeProblem,
      x0: _typing.PyTree,
      *,
      dual0: _typing.PyTree | None = None,
      jit: bool = True,
      callback: _typing.CallbackFn[IALMState] | None = None,
      time_limit: float | None = None,
  ) -> IALMState:
    """Runs `IALM` and returns the full `IALMState`."""
    if not problem.is_constrained:
      raise ValueError(
          "IALM requires a constrained `CompositeProblem` (`constraints` "
          "must not be None); use `APD`, `AIPP`, or `proxax.solve` instead."
      )
    if not jit and time_limit is not None and time_limit < 0:
      raise ValueError("`time_limit` must be non-negative.")

    base_oracle = problem.oracle
    x_proj = base_oracle.domain_project(tree.as_float_tree(x0))
    dtype = tree.infer_dtype(x_proj)
    bound_c = problem.bind_constraints(x_proj)
    assert bound_c is not None

    mult0 = (
        bound_c.zero_dual(x_proj)
        if dual0 is None
        else tree.as_float_tree(dual0)
    )
    use_apd = self._uses_apd
    L0_base = jnp.asarray(self.lipschitz_init, dtype=dtype)
    extra_s_evals = jnp.asarray(0, dtype=jnp.int32)
    extra_g_evals = jnp.asarray(0, dtype=jnp.int32)
    extra_p_evals = jnp.asarray(0, dtype=jnp.int32)

    if use_apd:
      lam0 = jnp.asarray(1.0, dtype=dtype)
    elif self.lam_init is not None:
      lam0 = jnp.asarray(self.lam_init, dtype=dtype)
    elif base_oracle.curvature_lower is not None:
      m_val = jnp.asarray(base_oracle.curvature_lower, dtype=dtype)
      lam0 = jnp.where(
          m_val > 0.0,
          0.5 / jnp.maximum(m_val, jnp.asarray(1e-12, dtype=dtype)),
          10.0 / jnp.maximum(L0_base, jnp.asarray(1e-6, dtype=dtype)),
      )
    else:
      ref_base = certificates.refine(base_oracle, x_proj, L0_base, jit=jit)
      L0_base = ref_base.lipschitz_used
      lam0 = jnp.clip(
          2.0 / jnp.maximum(L0_base, jnp.asarray(1e-6, dtype=dtype)),
          jnp.asarray(self.lam_min, dtype=dtype),
          jnp.asarray(self.lam_max, dtype=dtype),
      )
      extra_s_evals = ref_base.smooth_evals
      extra_g_evals = ref_base.grad_evals
      extra_p_evals = ref_base.prox_evals

    c0 = jnp.asarray(self.penalty_init, dtype=dtype)
    pen0 = composite.PenalizedOracle(
        base=base_oracle,
        constraint=bound_c,
        multiplier=mult0,
        penalty=c0,
        dampen=self.dampen,
    )
    ref0 = certificates.refine(pen0, x_proj, L0_base, jit=jit)
    p0 = pen0.effective_dual(ref0.x_hat)
    p_bar0 = tree.scale(1.0 - self.dampen, mult0)
    al_feas0 = tree.norm(
        tree.scale(1.0 / c0, tree.sub(p0, p_bar0)),
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

    m_min_arr = jnp.asarray(self.lower_curvature_min, dtype=dtype)
    m_max_arr = jnp.asarray(self.lower_curvature_max, dtype=dtype)
    if self.lower_curvature_init is not None:
      m0 = jnp.clip(
          jnp.asarray(self.lower_curvature_init, dtype=dtype),
          m_min_arr,
          m_max_arr,
      )
    elif base_oracle.curvature_lower is not None:
      if base_oracle.curvature_lower > 0.0:
        m0 = jnp.clip(
            jnp.asarray(base_oracle.curvature_lower, dtype=dtype),
            m_min_arr,
            m_max_arr,
        )
      else:
        m0 = jnp.clip(jnp.maximum(target_opt, m_min_arr), m_min_arr, m_max_arr)
    else:
      m0 = jnp.clip(jnp.asarray(1.0, dtype=dtype), m_min_arr, m_max_arr)

    tiny = jnp.asarray(1e-30, dtype=dtype)
    kkt_score0 = jnp.maximum(
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

    init_state = IALMState(
        primal=ref0.x_hat,
        multiplier=mult0,
        best_x=ref0.x_hat,
        best_dual=p0,
        best_subgrad=ref0.v_hat,
        best_opt_res=opt_res0,
        best_feas_res=feas_res0,
        best_kkt_score=kkt_score0,
        best_obj=obj0,
        prev_feas_res=feas_res0,
        target_opt_tol=target_opt,
        target_feas_tol=target_feas,
        penalty=c0,
        lam=lam0,
        lower_curvature=m0,
        upper_curvature=ref0.lipschitz_used,
        outer_iters=jnp.asarray(0, dtype=jnp.int32),
        inner_iters=jnp.asarray(0, dtype=jnp.int32),
        smooth_evals=extra_s_evals + ref0.smooth_evals + 1,
        grad_evals=extra_g_evals + ref0.grad_evals,
        prox_evals=extra_p_evals + ref0.prox_evals,
        status=status0,
    )

    apd_solver = apd_mod.APD(
        opt_tol=self.opt_tol,
        rel_opt_tol=0.0,
        norm_kind=self.norm_kind,
        alpha=self.alpha,
        beta=self.beta,
        theta=self.theta,
        rho=self.rho,
        decay_curvature=self.decay_curvature,
        lower_curvature_min=self.lower_curvature_min,
        lower_curvature_max=self.lower_curvature_max,
        lipschitz_init=self.lipschitz_init,
        max_outer_iters=self.max_subproblem_iters,
        max_inner_iters=self.max_inner_iters,
    )
    aipp_solver = aipp_mod.AIPP(
        opt_tol=self.opt_tol,
        rel_opt_tol=0.0,
        norm_kind=self.norm_kind,
        sigma=self.sigma,
        lam_min=self.lam_min,
        lam_max=self.lam_max,
        adaptive_prox=self.adaptive_prox,
        lipschitz_init=self.lipschitz_init,
        max_outer_iters=self.max_subproblem_iters,
        max_inner_iters=self.max_inner_iters,
    )
    start_wall = (
        time.perf_counter() if (not jit and time_limit is not None) else 0.0
    )

    def _cond(st: IALMState) -> jax.Array:
      return st.status == running_code

    def _body(st: IALMState) -> IALMState:
      pen_oracle = composite.PenalizedOracle(
          base=base_oracle,
          constraint=bound_c,
          multiplier=st.multiplier,
          penalty=st.penalty,
          dampen=self.dampen,
      )
      rem_time: float | None = None
      if not jit and time_limit is not None:
        rem_time = max(0.0, time_limit - (time.perf_counter() - start_wall))

      if use_apd:
        apd_st = apd_solver.solve_oracle(
            pen_oracle,
            st.primal,
            lower_curvature_init=st.lower_curvature,
            upper_curvature_init=st.upper_curvature,
            opt_tol_override=st.target_opt_tol,
            jit=jit,
            time_limit=rem_time,
        )
        cand_x = apd_st.best_x
        cand_v = apd_st.best_subgrad
        cand_opt = apd_st.best_opt_res
        lam_next = st.lam
        m_next = apd_st.lower_curvature
        upper_used = apd_st.upper_curvature
        sub_in_iters = apd_st.inner_iters
        sub_s_evals = apd_st.smooth_evals
        sub_g_evals = apd_st.grad_evals
        sub_p_evals = apd_st.prox_evals
        sub_status = apd_st.status
      else:
        aipp_st = aipp_solver.solve_oracle(
            pen_oracle,
            st.primal,
            lipschitz_init=st.upper_curvature,
            lam_init=st.lam,
            opt_tol_override=st.target_opt_tol,
            jit=jit,
            time_limit=rem_time,
        )
        cand_x = aipp_st.best_x
        cand_v = aipp_st.best_subgrad
        cand_opt = aipp_st.best_opt_res
        lam_next = aipp_st.lam
        m_next = st.lower_curvature
        upper_used = aipp_st.lipschitz
        sub_in_iters = aipp_st.inner_iters
        sub_s_evals = aipp_st.smooth_evals
        sub_g_evals = aipp_st.grad_evals
        sub_p_evals = aipp_st.prox_evals
        sub_status = aipp_st.status

      # Certified dual p_hat in N_C(y_tilde) and exact complementarity/primal
      # residual A(x_hat) - y_tilde = (p_hat - p_bar) / c_k.
      cand_p = pen_oracle.effective_dual(cand_x)
      p_bar = tree.scale(1.0 - self.dampen, st.multiplier)
      al_feas = tree.norm(
          tree.scale(1.0 / st.penalty, tree.sub(cand_p, p_bar)),
          kind=self.norm_kind,
      )
      raw_feas = tree.norm(bound_c.residual(cand_x), kind=self.norm_kind)
      cand_feas = jnp.maximum(al_feas, raw_feas)
      cand_obj = base_oracle.value(cand_x)
      cand_score = jnp.maximum(
          cand_opt / jnp.maximum(st.target_opt_tol, tiny),
          cand_feas / jnp.maximum(st.target_feas_tol, tiny),
      )

      is_better = cand_score <= st.best_kkt_score
      best_x = jax.tree.map(
          lambda cx, bx: jnp.where(is_better, cx, bx),
          cand_x,
          st.best_x,
      )
      best_p = jax.tree.map(
          lambda cp, bp: jnp.where(is_better, cp, bp),
          cand_p,
          st.best_dual,
      )
      best_v = jax.tree.map(
          lambda cv, bv: jnp.where(is_better, cv, bv),
          cand_v,
          st.best_subgrad,
      )
      best_opt = jnp.where(is_better, cand_opt, st.best_opt_res)
      best_feas = jnp.where(is_better, cand_feas, st.best_feas_res)
      best_score = jnp.where(is_better, cand_score, st.best_kkt_score)
      best_obj = jnp.where(is_better, cand_obj, st.best_obj)

      # Dampened proximal dual step: prox_{(chi * c_k) sigma_C}(p_bar + chi * c
      # * A(cand_x)), which equals `cand_p` when `dual_step == 1.0`.
      if self.dual_step == 1.0:
        next_mult = cand_p
      else:
        _, next_mult = bound_c.penalty_and_dual(
            cand_x,
            multiplier=st.multiplier,
            penalty=self.dual_step * st.penalty,
            dampen=self.dampen,
        )
      if self.dual_bound is not None:
        m_norm = tree.norm(next_mult, kind="l2")
        b_arr = jnp.asarray(self.dual_bound, dtype=dtype)
        clip_factor = jnp.where(
            m_norm > b_arr,
            b_arr / jnp.maximum(m_norm, tiny),
            jnp.asarray(1.0, dtype=dtype),
        )
        next_mult = tree.scale(clip_factor, next_mult)

      # Conditional penalty growth: only increase c_k when feasibility did not
      # contract by `feas_decay`.
      grow_c = cand_feas > jnp.maximum(
          st.target_feas_tol,
          self.feas_decay * st.prev_feas_res,
      )
      c_grown = jnp.minimum(
          st.penalty * self.penalty_factor,
          jnp.asarray(self.penalty_max, dtype=dtype),
      )
      c_next = jnp.where(grow_c, c_grown, st.penalty)
      prev_feas_next = jnp.minimum(st.prev_feas_res, cand_feas)
      upper_next = upper_used * (c_next / st.penalty)
      next_outer = st.outer_iters + 1

      finite_ok = jnp.logical_and(
          tree.all_finite(best_x),
          jnp.logical_and(jnp.isfinite(best_opt), jnp.isfinite(best_feas)),
      )
      kkt_converged = jnp.logical_and(
          best_opt <= st.target_opt_tol,
          best_feas <= st.target_feas_tol,
      )
      timeout_code = jnp.asarray(
          int(base.SolveStatus.TIME_LIMIT_REACHED),
          dtype=jnp.int32,
      )
      ls_fail_code = jnp.asarray(
          int(base.SolveStatus.LINE_SEARCH_FAILED),
          dtype=jnp.int32,
      )
      next_status = jnp.where(
          jnp.logical_not(finite_ok),
          jnp.asarray(int(base.SolveStatus.NON_FINITE), dtype=jnp.int32),
          jnp.where(
              sub_status == timeout_code,
              timeout_code,
              jnp.where(
                  sub_status == ls_fail_code,
                  ls_fail_code,
                  jnp.where(
                      kkt_converged,
                      jnp.asarray(
                          int(base.SolveStatus.CONVERGED),
                          dtype=jnp.int32,
                      ),
                      jnp.where(
                          next_outer >= self.max_alm_iters,
                          jnp.asarray(
                              int(base.SolveStatus.MAX_ITER_REACHED),
                              dtype=jnp.int32,
                          ),
                          running_code,
                      ),
                  ),
              ),
          ),
      )

      return IALMState(
          primal=cand_x,
          multiplier=next_mult,
          best_x=best_x,
          best_dual=best_p,
          best_subgrad=best_v,
          best_opt_res=best_opt,
          best_feas_res=best_feas,
          best_kkt_score=best_score,
          best_obj=best_obj,
          prev_feas_res=prev_feas_next,
          target_opt_tol=st.target_opt_tol,
          target_feas_tol=st.target_feas_tol,
          penalty=c_next,
          lam=lam_next,
          lower_curvature=m_next,
          upper_curvature=upper_next,
          outer_iters=next_outer,
          inner_iters=st.inner_iters + sub_in_iters,
          smooth_evals=st.smooth_evals + sub_s_evals + 1,
          grad_evals=st.grad_evals + sub_g_evals,
          prox_evals=st.prox_evals + sub_p_evals,
          status=next_status,
      )

    def _on_timeout(st: IALMState) -> IALMState:
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
      callback: _typing.CallbackFn[IALMState] | None = None,
      time_limit: float | None = None,
  ) -> base.Solution:
    """Solves a constrained `CompositeProblem` using `IALM`."""
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
