"""Dampened Inexact Proximal Augmented Lagrangian (`iALM`) solver."""

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
from proxax.solvers.unconstrained import aipp as aipp_mod
from proxax.solvers.unconstrained import apd as apd_mod
from proxax.solvers.unconstrained import panoc as panoc_mod


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IALMState(base.ConstrainedTrackerMixin):
  """Outer Augmented Lagrangian iteration state of the `IALM` solver."""

  primal: _typing.PyTree
  multiplier: _typing.PyTree
  prev_feas_res: jax.Array
  penalty: jax.Array
  lam: jax.Array
  lower_curvature: jax.Array
  upper_curvature: jax.Array
  tracker: base.ConstrainedTracker


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
    if key not in ("apd", "pf_apd", "aipp", "panoc"):
      raise ValueError(
          f"Unsupported `subproblem_solver` {self.subproblem_solver!r}; "
          "expected 'apd', 'aipp', or 'panoc'."
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

  @property
  def _uses_panoc(self) -> bool:
    key = self.subproblem_solver.strip().lower().replace("-", "_")
    return key == "panoc"

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
    if time_limit is not None and time_limit < 0:
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
    use_panoc = self._uses_panoc
    L0_base = jnp.asarray(self.lipschitz_init, dtype=dtype)
    extra_s_evals = jnp.asarray(0, dtype=jnp.int32)
    extra_g_evals = jnp.asarray(0, dtype=jnp.int32)
    extra_p_evals = jnp.asarray(0, dtype=jnp.int32)

    if use_apd or use_panoc:
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

    tracker0 = base.ConstrainedTracker.from_init(
        ref0.x_hat,
        p0,
        ref0.v_hat,
        opt_res0,
        feas_res0,
        obj0,
        opt_tol=self.opt_tol,
        feas_tol=self.feas_tol,
        rel_opt_tol=self.rel_opt_tol,
        rel_feas_tol=self.rel_feas_tol,
        smooth_evals=extra_s_evals + ref0.smooth_evals + 1,
        grad_evals=extra_g_evals + ref0.grad_evals,
        prox_evals=extra_p_evals + ref0.prox_evals,
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
        m0 = jnp.clip(
            jnp.maximum(tracker0.target_opt_tol, m_min_arr),
            m_min_arr,
            m_max_arr,
        )
    else:
      m0 = jnp.clip(jnp.asarray(1.0, dtype=dtype), m_min_arr, m_max_arr)

    init_state = IALMState(
        primal=ref0.x_hat,
        multiplier=mult0,
        prev_feas_res=feas_res0,
        penalty=c0,
        lam=lam0,
        lower_curvature=m0,
        upper_curvature=ref0.lipschitz_used,
        tracker=tracker0,
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
    panoc_solver = panoc_mod.PANOC(
        opt_tol=self.opt_tol,
        rel_opt_tol=0.0,
        norm_kind=self.norm_kind,
        lipschitz_init=self.lipschitz_init,
        max_iters=self.max_subproblem_iters,
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
    panoc_solver = panoc_mod.PANOC(
        opt_tol=self.opt_tol,
        rel_opt_tol=0.0,
        norm_kind=self.norm_kind,
        lipschitz_init=self.lipschitz_init,
        max_iters=self.max_subproblem_iters,
    )
    start_wall = (
        _loop.start_clock(jit=jit) if time_limit is not None else 0.0
    )

    def _body(st: IALMState) -> IALMState:
      pen_oracle = composite.PenalizedOracle(
          base=base_oracle,
          constraint=bound_c,
          multiplier=st.multiplier,
          penalty=st.penalty,
          dampen=self.dampen,
      )
      rem_time = _loop.remaining_time(jit, time_limit, start_wall)

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
        sub_counts = apd_st.counts
        sub_status = apd_st.status
      elif use_panoc:
        panoc_st = panoc_solver.solve_oracle(
            pen_oracle,
            st.primal,
            lipschitz_init=st.upper_curvature,
            opt_tol_override=st.target_opt_tol,
            jit=jit,
            time_limit=rem_time,
        )
        cand_x = panoc_st.best_x
        cand_v = panoc_st.best_subgrad
        cand_opt = panoc_st.best_opt_res
        lam_next = st.lam
        m_next = st.lower_curvature
        upper_used = panoc_st.lipschitz
        sub_in_iters = panoc_st.outer_iters
        sub_counts = panoc_st.counts
        sub_status = panoc_st.status
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
        sub_counts = aipp_st.counts
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
        tiny = jnp.asarray(1e-30, dtype=dtype)
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

      tracker = st.tracker.update_incumbent(
          cand_x,
          cand_p,
          cand_v,
          cand_opt,
          cand_feas,
          cand_obj,
      ).advance(
          inner_inc=sub_in_iters,
          smooth_inc=1,
          counts_inc=sub_counts,
          max_iters=self.max_alm_iters,
          sub_status=sub_status,
      )

      return IALMState(
          primal=cand_x,
          multiplier=next_mult,
          prev_feas_res=prev_feas_next,
          penalty=c_next,
          lam=lam_next,
          lower_curvature=m_next,
          upper_curvature=upper_next,
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
    return fin.to_solution()
