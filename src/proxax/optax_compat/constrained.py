"""Optax constrained transformations (`qp_aipp`, `qp_apd`, `ialm`)."""

from __future__ import annotations

import dataclasses
import typing

import jax
import jax.numpy as jnp
import optax

from proxax import _typing
from proxax import certificates
from proxax import tree
from proxax.constraints import linear_op
from proxax.optax_compat._subproblems import ACGSubproblemCarry
from proxax.optax_compat._subproblems import ACGSubproblemMixin
from proxax.optax_compat._subproblems import ConstrainedOptaxTrackerMixin
from proxax.optax_compat._subproblems import PFAPDSubproblemCarry
from proxax.optax_compat._subproblems import PFAPDSubproblemMixin
from proxax.optax_compat._subproblems import _accepted_apd_curvatures
from proxax.optax_compat._subproblems import _check_aipp_rejection
from proxax.optax_compat._subproblems import _evaluate_apd_subproblem
from proxax.optax_compat._subproblems import _initial_lam
from proxax.optax_compat._subproblems import _initial_lower_curvature
from proxax.optax_compat._subproblems import _validate_apd_params
from proxax.optax_compat._trackers import ConstrainedOptaxTracker
from proxax.oracle import composite
from proxax.oracle import terms
from proxax.prox import base as prox_base
from proxax.solvers.inner import acg
from proxax.solvers.inner import pf_acg

_ZERO_PROX: typing.Final = prox_base.Zero()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class QPTransformState(ConstrainedOptaxTrackerMixin, ACGSubproblemMixin):
  """Single-inner-step state machine for `proxax.optax_compat.qp_aipp`."""

  tracker: ConstrainedOptaxTracker
  subproblem: ACGSubproblemCarry

  @property
  def penalty_iter(self) -> jax.Array:
    """Completed outer quadratic-penalty stages."""
    return self.tracker.stage_iter


def qp_aipp(
    *,
    constraints: linear_op.Constraint | _typing.PyTree,
    prox_op: prox_base.ProxOperator = _ZERO_PROX,
    opt_tol: float = 1e-5,
    feas_tol: float = 1e-5,
    norm_kind: str = "l2",
    penalty_init: float = 1.0,
    penalty_factor: float = 2.0,
    penalty_max: float = 1e12,
    sigma: float = 0.3,
    lower_curvature: float | None = None,
    upper_curvature: float | None = None,
    lam_init: float | None = None,
    lam_min: float = 1e-10,
    lam_max: float = 1e8,
    lam_increase: float = 1.25,
    lam_decrease: float = 0.5,
    adaptive_prox: bool | None = None,
    lipschitz_init: float = 1.0,
    max_aipp_iters: int = 250,
    max_inner_iters: int = 1000,
) -> optax.GradientTransformationExtraArgs:
  """Creates an Optax single-step `QP-AIPP` transformation for constrained NCO.

  Solves `min_x value_fn(x) + prox_op(x)  s.t.  A(x) in C` inside a standard
  `tx.init` / `tx.update` / `optax.apply_updates` loop. Each `tx.update` runs
  one inner `ACG` step while keeping `params` in `dom(h)` and updating
  `state.dual`, `state.subgrad`, `state.opt_residual`, and
  `state.feas_residual`.

  Args:
    constraints: `Constraint` or PyTree (e.g. `dict` / `tuple`) of `Constraint`
      instances defining `A(x) in C`.
    prox_op: Nonsmooth convex `ProxOperator` `h` (defaults to `Zero()`).
    opt_tol: Lagrangian stationarity tolerance `||v_hat|| <= opt_tol`.
    feas_tol: Primal feasibility tolerance `||A(x_hat) - Pi_C(A(x_hat))||`.
    norm_kind: Residual norm (`'l2'` or `'linf'`).
    penalty_init: Initial quadratic penalty parameter `c_0 > 0`.
    penalty_factor: Multiplicative penalty growth factor (`> 1`).
    penalty_max: Maximum allowed penalty parameter `c_max`.
    sigma: Inner `ACG` relative proximal error parameter in `(0, 1)`.
    lower_curvature: Optional weak-convexity constant `m >= 0` of `value_fn`.
    upper_curvature: Optional global smoothness constant `L^+` of `value_fn`.
    lam_init: Optional initial proximal step size `lambda_0 > 0`.
    lam_min: Minimum allowed proximal step size.
    lam_max: Maximum allowed proximal step size.
    lam_increase: Growth factor for `lam` after fast inner convergence.
    lam_decrease: Shrink factor (`< 1`) when nonconvexity is detected.
    adaptive_prox: Whether to adapt `lam` (`None` = automatic).
    lipschitz_init: Initial smoothness estimate `L_0 > 0` for `value_fn`.
    max_aipp_iters: Maximum `AIPP` proximal steps per penalty stage.
    max_inner_iters: Maximum `ACG` steps per proximal subproblem.

  Returns:
    An `optax.GradientTransformationExtraArgs`.
  """
  norm_c = linear_op.normalize_constraints(constraints)
  if norm_c is None:
    raise ValueError(
        "`qp_aipp` requires non-empty `constraints`; use `aipp` for "
        "unconstrained problems."
    )
  has_known_m = lower_curvature is not None
  use_adaptive = (not has_known_m) if adaptive_prox is None else adaptive_prox
  acg_cfg = acg.ACGConfig(sigma=sigma, max_iters=max_inner_iters)

  def init_fn(params: _typing.PyTree) -> QPTransformState:
    p0 = prox_op.domain_project(tree.as_float_tree(params))
    dtype = tree.infer_dtype(p0)
    bound_c = norm_c.bind(p0)
    dual0 = bound_c.zero_dual(p0)
    lam0 = _initial_lam(
        lam_init=lam_init,
        lower_curvature=lower_curvature,
        lipschitz_init=lipschitz_init,
        dtype=dtype,
    )
    return QPTransformState(
        tracker=ConstrainedOptaxTracker.uninitialized(
            p0,
            dual0=dual0,
            penalty_init=penalty_init,
        ),
        subproblem=ACGSubproblemCarry.create(
            p0,
            jnp.asarray(0.0, dtype=dtype),
            lam=lam0,
            lipschitz=jnp.asarray(lipschitz_init, dtype=dtype),
        ),
    )

  def update_fn(
      updates: _typing.PyTree,
      state: QPTransformState,
      params: _typing.PyTree | None = None,
      *,
      value: jax.Array | None = None,
      value_fn: _typing.SmoothFn | None = None,
      **extra_args: typing.Any,
  ) -> tuple[_typing.PyTree, QPTransformState]:
    del updates, value, extra_args
    if params is None:
      raise ValueError(
          "`proxax.optax_compat.qp_aipp` requires `params` in `tx.update`."
      )
    if value_fn is None:
      raise ValueError(
          "`proxax.optax_compat.qp_aipp` requires "
          "`value_fn=<smooth_objective>` in `tx.update`."
      )

    smooth_term = terms.as_smooth_term(
        value_fn,
        upper_curvature=upper_curvature,
        lower_curvature=lower_curvature,
    )
    base_oracle = composite.CompositeOracle(
        smooth=smooth_term,
        proximal=prox_op,
    )
    bound_c = norm_c.bind(state.center)
    dtype = state.lipschitz.dtype
    opt_tol_arr = jnp.asarray(opt_tol, dtype=dtype)
    feas_tol_arr = jnp.asarray(feas_tol, dtype=dtype)

    def _bootstrap(st: QPTransformState) -> QPTransformState:
      p_proj = prox_op.domain_project(tree.as_float_tree(params))
      if lam_init is None and not has_known_m:
        ref_base = certificates.refine(
            base_oracle,
            p_proj,
            st.lipschitz,
            jit=True,
        )
        lam0 = jnp.clip(
            2.0
            / jnp.maximum(
                ref_base.lipschitz_used,
                jnp.asarray(1e-6, dtype=dtype),
            ),
            jnp.asarray(lam_min, dtype=dtype),
            jnp.asarray(lam_max, dtype=dtype),
        )
        L_seed = ref_base.lipschitz_used
      else:
        lam0 = st.lam
        L_seed = st.lipschitz

      pen0 = composite.PenalizedOracle(
          base=base_oracle,
          constraint=bound_c,
          multiplier=None,
          penalty=st.penalty,
          dampen=0.0,
      )
      ref0 = certificates.refine(pen0, p_proj, L_seed, jit=True)
      d0 = pen0.effective_dual(ref0.x_hat)
      opt0 = tree.norm(ref0.v_hat, kind=norm_kind)
      feas0 = tree.norm(bound_c.residual(ref0.x_hat), kind=norm_kind)
      conv0 = jnp.logical_and(opt0 <= opt_tol_arr, feas0 <= feas_tol_arr)
      return QPTransformState(
          tracker=st.tracker.bootstrap(
              center=ref0.x_hat,
              center_obj=ref0.obj_val,
              dual=d0,
              subgrad=ref0.v_hat,
              opt_residual=opt0,
              feas_residual=feas0,
              converged=conv0,
          ),
          subproblem=ACGSubproblemCarry.create(
              ref0.x_hat,
              ref0.obj_val,
              lam=lam0,
              lipschitz=ref0.lipschitz_used,
          ),
      )

    st_ready = jax.lax.cond(
        state.initialized,
        lambda s: s,
        _bootstrap,
        state,
    )

    def _when_converged(st: QPTransformState) -> QPTransformState:
      return dataclasses.replace(st, tracker=st.tracker.increment_step())

    def _when_active(st: QPTransformState) -> QPTransformState:
      pen_oracle = composite.PenalizedOracle(
          base=base_oracle,
          constraint=bound_c,
          multiplier=None,
          penalty=st.penalty,
          dampen=0.0,
      )
      acg_out, smooth_reg, inv_lam = st.subproblem.step(
          pen_oracle,
          center=st.center,
          inner_iter=st.inner_iter,
          config=acg_cfg,
          has_known_m=has_known_m,
          lower_curvature=lower_curvature,
          use_adaptive=use_adaptive,
      )
      running_code = jnp.asarray(int(acg.ACGStatus.RUNNING), dtype=jnp.int32)
      still_running = acg_out.status == running_code

      def _continue_inner() -> QPTransformState:
        return QPTransformState(
            tracker=dataclasses.replace(
                st.tracker,
                inner_iter=acg_out.iteration,
                total_steps=st.total_steps + 1,
            ),
            subproblem=st.subproblem.continue_from(
                acg_out, smooth_reg=smooth_reg
            ),
        )

      def _finish_subproblem() -> QPTransformState:
        phi_y = pen_oracle.value(acg_out.y)
        rejected = _check_aipp_rejection(
            acg_out,
            center=st.center,
            center_obj=st.center_obj,
            phi_y=phi_y,
            lam=st.lam,
            inv_lam=inv_lam,
            sigma=sigma,
            use_adaptive=use_adaptive,
        )

        def _reject_outer() -> QPTransformState:
          shrunk = jnp.maximum(
              st.lam * lam_decrease,
              jnp.asarray(lam_min, dtype=dtype),
          )
          est_L = jnp.maximum(acg_out.lipschitz - smooth_reg, st.lipschitz)
          return QPTransformState(
              tracker=dataclasses.replace(
                  st.tracker,
                  stage_outer_iter=st.stage_outer_iter + 1,
                  outer_iter=st.outer_iter + 1,
                  inner_iter=jnp.asarray(0, dtype=jnp.int32),
                  total_steps=st.total_steps + 1,
              ),
              subproblem=ACGSubproblemCarry.create(
                  st.center,
                  st.sub_psi_x0,
                  lam=shrunk,
                  lipschitz=est_L,
              ),
          )

        def _accept_outer() -> QPTransformState:
          est_L = jnp.maximum(
              acg_out.lipschitz - smooth_reg,
              0.5 * st.lipschitz,
          )
          ref = certificates.refine(pen_oracle, acg_out.y, est_L, jit=True)
          cand_opt = tree.norm(ref.v_hat, kind=norm_kind)
          cand_feas = tree.norm(bound_c.residual(ref.x_hat), kind=norm_kind)
          cand_dual = pen_oracle.effective_dual(ref.x_hat)
          conv = jnp.logical_and(
              cand_opt <= opt_tol_arr,
              cand_feas <= feas_tol_arr,
          )

          next_stage_outer = st.stage_outer_iter + 1
          advance_penalty = jnp.logical_and(
              jnp.logical_not(conv),
              jnp.logical_or(
                  cand_opt <= opt_tol_arr,
                  next_stage_outer >= max_aipp_iters,
              ),
          )

          use_ref = ref.obj_val <= phi_y
          cand_center = jax.tree.map(
              lambda rx, yx: jnp.where(use_ref, rx, yx),
              ref.x_hat,
              acg_out.y,
          )
          next_center = jax.tree.map(
              lambda rx, cc: jnp.where(conv, rx, cc),
              ref.x_hat,
              cand_center,
          )

          c_grown = jnp.minimum(
              st.penalty * penalty_factor,
              jnp.asarray(penalty_max, dtype=dtype),
          )
          next_penalty = jnp.where(advance_penalty, c_grown, st.penalty)
          next_pen_oracle = composite.PenalizedOracle(
              base=base_oracle,
              constraint=bound_c,
              multiplier=None,
              penalty=next_penalty,
              dampen=0.0,
          )
          next_center_obj = next_pen_oracle.value(next_center)
          next_L = jnp.where(
              advance_penalty,
              ref.lipschitz_used * (next_penalty / st.penalty),
              ref.lipschitz_used,
          )
          grow = jnp.logical_and(
              jnp.asarray(use_adaptive),
              acg_out.iteration <= 8,
          )
          next_lam = jnp.where(
              grow,
              jnp.minimum(
                  st.lam * lam_increase,
                  jnp.asarray(lam_max, dtype=dtype),
              ),
              st.lam,
          )
          return QPTransformState(
              tracker=ConstrainedOptaxTracker(
                  initialized=jnp.asarray(True, dtype=jnp.bool_),
                  converged=conv,
                  stage_iter=jnp.where(
                      advance_penalty,
                      st.stage_iter + 1,
                      st.stage_iter,
                  ),
                  stage_outer_iter=jnp.where(
                      advance_penalty,
                      jnp.asarray(0, dtype=jnp.int32),
                      next_stage_outer,
                  ),
                  outer_iter=st.outer_iter + 1,
                  inner_iter=jnp.asarray(0, dtype=jnp.int32),
                  total_steps=st.total_steps + 1,
                  center=next_center,
                  center_obj=next_center_obj,
                  best_x=ref.x_hat,
                  penalty=next_penalty,
                  dual=cand_dual,
                  subgrad=ref.v_hat,
                  opt_residual=cand_opt,
                  feas_residual=cand_feas,
              ),
              subproblem=ACGSubproblemCarry.create(
                  next_center,
                  next_center_obj,
                  lam=next_lam,
                  lipschitz=next_L,
              ),
          )

        return typing.cast(
            QPTransformState,
            jax.lax.cond(rejected, _reject_outer, _accept_outer),
        )

      return typing.cast(
          QPTransformState,
          jax.lax.cond(still_running, _continue_inner, _finish_subproblem),
      )

    next_state = typing.cast(
        QPTransformState,
        jax.lax.cond(
            st_ready.converged,
            _when_converged,
            _when_active,
            st_ready,
        ),
    )
    target_primal = jax.tree.map(
        lambda bx, ay: jnp.where(next_state.converged, bx, ay),
        next_state.best_x,
        next_state.acg_y,
    )
    return tree.sub(target_primal, params), next_state

  return optax.GradientTransformationExtraArgs(init=init_fn, update=update_fn)




@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class QPAPDTransformState(ConstrainedOptaxTrackerMixin, PFAPDSubproblemMixin):
  """Single-inner-step state machine for `proxax.optax_compat.qp_apd`."""

  tracker: ConstrainedOptaxTracker
  subproblem: PFAPDSubproblemCarry

  @property
  def penalty_iter(self) -> jax.Array:
    """Completed outer quadratic-penalty stages."""
    return self.tracker.stage_iter


def qp_apd(
    *,
    constraints: linear_op.Constraint | _typing.PyTree,
    prox_op: prox_base.ProxOperator = _ZERO_PROX,
    opt_tol: float = 1e-5,
    feas_tol: float = 1e-5,
    norm_kind: str = "l2",
    penalty_init: float = 1.0,
    penalty_factor: float = 2.0,
    penalty_max: float = 1e12,
    alpha: float = 2.0,
    beta: float = 2.0,
    theta: float = 4.0,
    rho: float | None = None,
    decay_curvature: bool = True,
    lower_curvature: float | None = None,
    upper_curvature: float | None = None,
    lower_curvature_init: float | None = None,
    lower_curvature_min: float = 1e-6,
    lower_curvature_max: float = 1e12,
    lipschitz_init: float = 1.0,
    max_apd_iters: int = 250,
    max_inner_iters: int = 1000,
) -> optax.GradientTransformationExtraArgs:
  """Creates an Optax single-step `QP-APD` transformation for constrained NCO.

  Solves `min_x value_fn(x) + prox_op(x)  s.t.  A(x) in C` inside a standard
  `tx.init` / `tx.update` / `optax.apply_updates` loop. Each `tx.update` runs
  one inner `PF.ACG` step with refinement-free certificates while updating
  `state.dual`, `state.subgrad`, `state.opt_residual`, and
  `state.feas_residual`.

  Args:
    constraints: `Constraint` or PyTree of `Constraint` instances defining
      `A(x) in C`.
    prox_op: Nonsmooth convex `ProxOperator` `h` (defaults to `Zero()`).
    opt_tol: Lagrangian stationarity tolerance `||v_hat|| <= opt_tol`.
    feas_tol: Primal feasibility tolerance `||A(x_hat) - Pi_C(A(x_hat))||`.
    norm_kind: Residual norm (`'l2'` or `'linf'`).
    penalty_init: Initial quadratic penalty parameter `c_0 > 0`.
    penalty_factor: Multiplicative penalty growth factor (`> 1`).
    penalty_max: Maximum allowed penalty parameter `c_max`.
    alpha: Multiplicative growth factor (`> 1`) for lower curvature `m_hat`.
    beta: Multiplicative growth factor (`> 1`) for inner smoothness `L`.
    theta: Proximal descent constant (`> 2`, default `4.0`).
    rho: Optional inner relative proximal error tolerance in `(0, 1)`.
    decay_curvature: Whether to apply practical curvature decay Eq. (6.1).
    lower_curvature: Optional weak-convexity constant `m >= 0` of `value_fn`.
    upper_curvature: Optional global smoothness constant `M` of `value_fn`.
    lower_curvature_init: Optional initial lower curvature `m_0 > 0` for `f`.
    lower_curvature_min: Floor `m_min > 0` guarding curvature decay.
    lower_curvature_max: Maximum allowed lower curvature estimate.
    lipschitz_init: Initial smoothness estimate `M_0 > 0` for `value_fn`.
    max_apd_iters: Maximum `APD` proximal steps per penalty stage.
    max_inner_iters: Maximum `PF.ACG` steps per proximal subproblem.

  Returns:
    An `optax.GradientTransformationExtraArgs`.
  """
  norm_c = linear_op.normalize_constraints(constraints)
  if norm_c is None:
    raise ValueError(
        "`qp_apd` requires non-empty `constraints`; use `apd` for "
        "unconstrained problems."
    )
  eff_rho, pf_cfg = _validate_apd_params(
      alpha=alpha,
      beta=beta,
      theta=theta,
      rho=rho,
      decay_curvature=decay_curvature,
      max_inner_iters=max_inner_iters,
  )

  def init_fn(params: _typing.PyTree) -> QPAPDTransformState:
    p0 = prox_op.domain_project(tree.as_float_tree(params))
    dtype = tree.infer_dtype(p0)
    bound_c = norm_c.bind(p0)
    dual0 = bound_c.zero_dual(p0)
    m0 = _initial_lower_curvature(
        lower_curvature_init=lower_curvature_init,
        lower_curvature=lower_curvature,
        opt_tol=opt_tol,
        lower_curvature_min=lower_curvature_min,
        lower_curvature_max=lower_curvature_max,
        dtype=dtype,
    )
    M0 = jnp.asarray(lipschitz_init, dtype=dtype)
    return QPAPDTransformState(
        tracker=ConstrainedOptaxTracker.uninitialized(
            p0,
            dual0=dual0,
            penalty_init=penalty_init,
        ),
        subproblem=PFAPDSubproblemCarry.create(
            p0,
            jnp.asarray(0.0, dtype=dtype),
            lower_curvature=m0,
            upper_curvature=M0,
            never_increased_m=jnp.asarray(True, dtype=jnp.bool_),
            decay_curvature=decay_curvature,
            beta=beta,
        ),
    )

  def update_fn(
      updates: _typing.PyTree,
      state: QPAPDTransformState,
      params: _typing.PyTree | None = None,
      *,
      value: jax.Array | None = None,
      value_fn: _typing.SmoothFn | None = None,
      **extra_args: typing.Any,
  ) -> tuple[_typing.PyTree, QPAPDTransformState]:
    del updates, value, extra_args
    if params is None:
      raise ValueError(
          "`proxax.optax_compat.qp_apd` requires `params` in `tx.update`."
      )
    if value_fn is None:
      raise ValueError(
          "`proxax.optax_compat.qp_apd` requires "
          "`value_fn=<smooth_objective>` in `tx.update`."
      )

    smooth_term = terms.as_smooth_term(
        value_fn,
        upper_curvature=upper_curvature,
        lower_curvature=lower_curvature,
    )
    base_oracle = composite.CompositeOracle(
        smooth=smooth_term,
        proximal=prox_op,
    )
    bound_c = norm_c.bind(state.center)
    dtype = state.upper_curvature.dtype
    opt_tol_arr = jnp.asarray(opt_tol, dtype=dtype)
    feas_tol_arr = jnp.asarray(feas_tol, dtype=dtype)
    m_min_arr = jnp.asarray(lower_curvature_min, dtype=dtype)
    m_max_arr = jnp.asarray(lower_curvature_max, dtype=dtype)

    def _bootstrap(st: QPAPDTransformState) -> QPAPDTransformState:
      p_proj = prox_op.domain_project(tree.as_float_tree(params))
      pen0 = composite.PenalizedOracle(
          base=base_oracle,
          constraint=bound_c,
          multiplier=None,
          penalty=st.penalty,
          dampen=0.0,
      )
      ref0 = certificates.refine(pen0, p_proj, st.upper_curvature, jit=True)
      d0 = pen0.effective_dual(ref0.x_hat)
      opt0 = tree.norm(ref0.v_hat, kind=norm_kind)
      feas0 = tree.norm(bound_c.residual(ref0.x_hat), kind=norm_kind)
      conv0 = jnp.logical_and(opt0 <= opt_tol_arr, feas0 <= feas_tol_arr)
      M0 = ref0.lipschitz_used
      return QPAPDTransformState(
          tracker=st.tracker.bootstrap(
              center=ref0.x_hat,
              center_obj=ref0.obj_val,
              dual=d0,
              subgrad=ref0.v_hat,
              opt_residual=opt0,
              feas_residual=feas0,
              converged=conv0,
          ),
          subproblem=PFAPDSubproblemCarry.create(
              ref0.x_hat,
              ref0.obj_val,
              lower_curvature=st.lower_curvature,
              upper_curvature=M0,
              never_increased_m=st.never_increased_m,
              decay_curvature=decay_curvature,
              beta=beta,
          ),
      )

    st_ready = jax.lax.cond(
        state.initialized,
        lambda s: s,
        _bootstrap,
        state,
    )

    def _when_converged(st: QPAPDTransformState) -> QPAPDTransformState:
      return dataclasses.replace(st, tracker=st.tracker.increment_step())

    def _when_active(st: QPAPDTransformState) -> QPAPDTransformState:
      pen_oracle = composite.PenalizedOracle(
          base=base_oracle,
          constraint=bound_c,
          multiplier=None,
          penalty=st.penalty,
          dampen=0.0,
      )
      m_hat = st.lower_curvature
      pf_out, u_cand, v_cand, cand_opt = st.subproblem.step(
          pen_oracle,
          center=st.center,
          inner_iter=st.inner_iter,
          config=pf_cfg,
          opt_tol=opt_tol_arr,
          norm_kind=norm_kind,
      )
      cand_feas = tree.norm(bound_c.residual(pf_out.y), kind=norm_kind)
      cand_dual = pen_oracle.effective_dual(pf_out.y)
      upd_tr = st.tracker.update_incumbent(
          pf_out.y,
          cand_dual,
          v_cand,
          cand_opt,
          cand_feas,
          opt_tol_arr=opt_tol_arr,
          feas_tol_arr=feas_tol_arr,
      )
      conv_now = jnp.logical_and(
          upd_tr.opt_residual <= opt_tol_arr,
          upd_tr.feas_residual <= feas_tol_arr,
      )
      running_code = jnp.asarray(
          int(pf_acg.PFACGStatus.RUNNING), dtype=jnp.int32
      )
      still_running = jnp.logical_and(
          pf_out.status == running_code,
          jnp.logical_not(conv_now),
      )

      def _continue_inner() -> QPAPDTransformState:
        return QPAPDTransformState(
            tracker=dataclasses.replace(
                upd_tr,
                inner_iter=pf_out.iteration,
                total_steps=st.total_steps + 1,
            ),
            subproblem=st.subproblem.continue_from(pf_out),
        )

      def _finish_subproblem() -> QPAPDTransformState:
        phi_y = pen_oracle.value(pf_out.y)
        stage_opt_done = cand_opt <= opt_tol_arr
        rejected, dist_sq, M_from_acg = _evaluate_apd_subproblem(
            pf_out,
            center=st.center,
            center_obj=st.center_obj,
            phi_y=phi_y,
            u_cand=u_cand,
            v_cand=v_cand,
            m_hat=m_hat,
            theta=theta,
            eff_rho=eff_rho,
            done_flag=jnp.logical_or(conv_now, stage_opt_done),
        )

        def _reject_outer() -> QPAPDTransformState:
          grown_m = jnp.minimum(m_hat * alpha, m_max_arr)
          M_keep = jnp.maximum(st.upper_curvature, M_from_acg)
          return QPAPDTransformState(
              tracker=dataclasses.replace(
                  upd_tr,
                  stage_outer_iter=st.stage_outer_iter + 1,
                  outer_iter=st.outer_iter + 1,
                  inner_iter=jnp.asarray(0, dtype=jnp.int32),
                  total_steps=st.total_steps + 1,
              ),
              subproblem=PFAPDSubproblemCarry.create(
                  st.center,
                  st.center_obj,
                  lower_curvature=grown_m,
                  upper_curvature=M_keep,
                  never_increased_m=jnp.asarray(False, dtype=jnp.bool_),
                  decay_curvature=decay_curvature,
                  beta=beta,
              ),
          )

        def _accept_outer() -> QPAPDTransformState:
          m_next, M_est = _accepted_apd_curvatures(
              m_hat=m_hat,
              upper_curvature=st.upper_curvature,
              never_increased_m=st.never_increased_m,
              M_from_acg=M_from_acg,
              dist_sq=dist_sq,
              phi_y=phi_y,
              alpha=alpha,
              decay_curvature=decay_curvature,
              m_min_arr=m_min_arr,
          )
          next_stage_outer = st.stage_outer_iter + 1
          advance_penalty = jnp.logical_and(
              jnp.logical_not(conv_now),
              jnp.logical_or(
                  stage_opt_done,
                  next_stage_outer >= max_apd_iters,
              ),
          )
          c_grown = jnp.minimum(
              st.penalty * penalty_factor,
              jnp.asarray(penalty_max, dtype=dtype),
          )
          next_penalty = jnp.where(advance_penalty, c_grown, st.penalty)
          M_next = jnp.where(
              advance_penalty,
              M_est * (next_penalty / st.penalty),
              M_est,
          )
          next_center = jax.tree.map(
              lambda bx, py: jnp.where(conv_now, bx, py),
              upd_tr.best_x,
              pf_out.y,
          )
          next_pen_oracle = composite.PenalizedOracle(
              base=base_oracle,
              constraint=bound_c,
              multiplier=None,
              penalty=next_penalty,
              dampen=0.0,
          )
          next_center_obj = jnp.where(
              jnp.logical_or(advance_penalty, conv_now),
              next_pen_oracle.value(next_center),
              phi_y,
          )
          return QPAPDTransformState(
              tracker=dataclasses.replace(
                  upd_tr,
                  converged=conv_now,
                  stage_iter=jnp.where(
                      advance_penalty,
                      st.stage_iter + 1,
                      st.stage_iter,
                  ),
                  stage_outer_iter=jnp.where(
                      advance_penalty,
                      jnp.asarray(0, dtype=jnp.int32),
                      next_stage_outer,
                  ),
                  outer_iter=st.outer_iter + 1,
                  inner_iter=jnp.asarray(0, dtype=jnp.int32),
                  total_steps=st.total_steps + 1,
                  center=next_center,
                  center_obj=next_center_obj,
                  penalty=next_penalty,
              ),
              subproblem=PFAPDSubproblemCarry.create(
                  next_center,
                  next_center_obj,
                  lower_curvature=m_next,
                  upper_curvature=M_next,
                  never_increased_m=st.never_increased_m,
                  decay_curvature=decay_curvature,
                  beta=beta,
              ),
          )

        return typing.cast(
            QPAPDTransformState,
            jax.lax.cond(rejected, _reject_outer, _accept_outer),
        )

      return typing.cast(
          QPAPDTransformState,
          jax.lax.cond(still_running, _continue_inner, _finish_subproblem),
      )

    next_state = typing.cast(
        QPAPDTransformState,
        jax.lax.cond(
            st_ready.converged,
            _when_converged,
            _when_active,
            st_ready,
        ),
    )
    target_primal = jax.tree.map(
        lambda bx, py: jnp.where(next_state.converged, bx, py),
        next_state.best_x,
        next_state.pf_y,
    )
    return tree.sub(target_primal, params), next_state

  return optax.GradientTransformationExtraArgs(init=init_fn, update=update_fn)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IALMTransformState(ConstrainedOptaxTrackerMixin, PFAPDSubproblemMixin):
  """Single-inner-step state machine for `proxax.optax_compat.ialm`."""

  multiplier: _typing.PyTree
  prev_feas_res: jax.Array
  tracker: ConstrainedOptaxTracker
  subproblem: PFAPDSubproblemCarry

  @property
  def alm_iter(self) -> jax.Array:
    """Completed outer Augmented Lagrangian stages."""
    return self.tracker.stage_iter


def ialm(
    *,
    constraints: linear_op.Constraint | _typing.PyTree,
    prox_op: prox_base.ProxOperator = _ZERO_PROX,
    opt_tol: float = 1e-5,
    feas_tol: float = 1e-5,
    norm_kind: str = "l2",
    dampen: float = 0.0,
    dual_step: float = 1.0,
    penalty_init: float = 1.0,
    penalty_factor: float = 2.0,
    penalty_max: float = 1e12,
    feas_decay: float = 0.5,
    alpha: float = 2.0,
    beta: float = 2.0,
    theta: float = 4.0,
    rho: float | None = None,
    decay_curvature: bool = True,
    lower_curvature: float | None = None,
    upper_curvature: float | None = None,
    lower_curvature_init: float | None = None,
    lower_curvature_min: float = 1e-6,
    lower_curvature_max: float = 1e12,
    lipschitz_init: float = 1.0,
    max_apd_iters: int = 250,
    max_inner_iters: int = 1000,
) -> optax.GradientTransformationExtraArgs:
  """Creates an Optax single-step dampened `iALM` transformation.

  Solves `min_x value_fn(x) + prox_op(x)  s.t.  A(x) in C` via parameter-free
  `APD` Augmented Lagrangian stages (`dampen=0.0` for undampened `iALM` and
  `dampen` in `(0, 1)` for `AIDAL`). Each `tx.update` executes one inner
  `PF.ACG` step.

  Args:
    constraints: `Constraint` or PyTree of `Constraint` instances defining
      `A(x) in C`.
    prox_op: Nonsmooth convex `ProxOperator` `h` (defaults to `Zero()`).
    opt_tol: Lagrangian stationarity tolerance `||v_hat|| <= opt_tol`.
    feas_tol: Primal feasibility and complementarity tolerance.
    norm_kind: Residual norm (`'l2'` or `'linf'`).
    dampen: Multiplier dampening factor in `[0, 1)`.
    dual_step: Dual update step fraction in `(0, 1]`.
    penalty_init: Initial quadratic penalty parameter `c_0 > 0`.
    penalty_factor: Multiplicative penalty growth factor (`> 1`).
    penalty_max: Maximum allowed penalty parameter `c_max`.
    feas_decay: Required feasibility contraction factor in `[0, 1)` to keep
      `c_k` fixed.
    alpha: Multiplicative growth factor (`> 1`) for lower curvature `m_hat`.
    beta: Multiplicative growth factor (`> 1`) for inner smoothness `L`.
    theta: Proximal descent constant (`> 2`, default `4.0`).
    rho: Optional inner relative proximal error tolerance in `(0, 1)`.
    decay_curvature: Whether to apply practical curvature decay Eq. (6.1).
    lower_curvature: Optional weak-convexity constant `m >= 0` of `value_fn`.
    upper_curvature: Optional global smoothness constant `M` of `value_fn`.
    lower_curvature_init: Optional initial lower curvature `m_0 > 0` for `f`.
    lower_curvature_min: Floor `m_min > 0` guarding curvature decay.
    lower_curvature_max: Maximum allowed lower curvature estimate.
    lipschitz_init: Initial smoothness estimate `M_0 > 0` for `value_fn`.
    max_apd_iters: Maximum `APD` proximal steps per AL stage.
    max_inner_iters: Maximum `PF.ACG` steps per proximal subproblem.

  Returns:
    An `optax.GradientTransformationExtraArgs`.
  """
  norm_c = linear_op.normalize_constraints(constraints)
  if norm_c is None:
    raise ValueError(
        "`ialm` requires non-empty `constraints`; use `apd` for "
        "unconstrained problems."
    )
  if not (0.0 <= dampen < 1.0):
    raise ValueError("`dampen` must lie in `[0, 1)`.")
  if not (0.0 < dual_step <= 1.0):
    raise ValueError("`dual_step` must lie in `(0, 1]`.")
  if penalty_factor <= 1.0:
    raise ValueError("`penalty_factor` must be strictly greater than 1.")
  if not (0.0 <= feas_decay < 1.0):
    raise ValueError("`feas_decay` must lie in `[0, 1)`.")
  eff_rho, pf_cfg = _validate_apd_params(
      alpha=alpha,
      beta=beta,
      theta=theta,
      rho=rho,
      decay_curvature=decay_curvature,
      max_inner_iters=max_inner_iters,
  )

  def init_fn(params: _typing.PyTree) -> IALMTransformState:
    p0 = prox_op.domain_project(tree.as_float_tree(params))
    dtype = tree.infer_dtype(p0)
    bound_c = norm_c.bind(p0)
    dual0 = bound_c.zero_dual(p0)
    m0 = _initial_lower_curvature(
        lower_curvature_init=lower_curvature_init,
        lower_curvature=lower_curvature,
        opt_tol=opt_tol,
        lower_curvature_min=lower_curvature_min,
        lower_curvature_max=lower_curvature_max,
        dtype=dtype,
    )
    M0 = jnp.asarray(lipschitz_init, dtype=dtype)
    return IALMTransformState(
        multiplier=dual0,
        prev_feas_res=jnp.asarray(jnp.inf, dtype=dtype),
        tracker=ConstrainedOptaxTracker.uninitialized(
            p0,
            dual0=dual0,
            penalty_init=penalty_init,
        ),
        subproblem=PFAPDSubproblemCarry.create(
            p0,
            jnp.asarray(0.0, dtype=dtype),
            lower_curvature=m0,
            upper_curvature=M0,
            never_increased_m=jnp.asarray(True, dtype=jnp.bool_),
            decay_curvature=decay_curvature,
            beta=beta,
        ),
    )

  def update_fn(
      updates: _typing.PyTree,
      state: IALMTransformState,
      params: _typing.PyTree | None = None,
      *,
      value: jax.Array | None = None,
      value_fn: _typing.SmoothFn | None = None,
      **extra_args: typing.Any,
  ) -> tuple[_typing.PyTree, IALMTransformState]:
    del updates, value, extra_args
    if params is None:
      raise ValueError(
          "`proxax.optax_compat.ialm` requires `params` in `tx.update`."
      )
    if value_fn is None:
      raise ValueError(
          "`proxax.optax_compat.ialm` requires "
          "`value_fn=<smooth_objective>` in `tx.update`."
      )

    smooth_term = terms.as_smooth_term(
        value_fn,
        upper_curvature=upper_curvature,
        lower_curvature=lower_curvature,
    )
    base_oracle = composite.CompositeOracle(
        smooth=smooth_term,
        proximal=prox_op,
    )
    bound_c = norm_c.bind(state.center)
    dtype = state.upper_curvature.dtype
    opt_tol_arr = jnp.asarray(opt_tol, dtype=dtype)
    feas_tol_arr = jnp.asarray(feas_tol, dtype=dtype)
    m_min_arr = jnp.asarray(lower_curvature_min, dtype=dtype)
    m_max_arr = jnp.asarray(lower_curvature_max, dtype=dtype)

    def _bootstrap(st: IALMTransformState) -> IALMTransformState:
      p_proj = prox_op.domain_project(tree.as_float_tree(params))
      pen0 = composite.PenalizedOracle(
          base=base_oracle,
          constraint=bound_c,
          multiplier=st.multiplier,
          penalty=st.penalty,
          dampen=dampen,
      )
      ref0 = certificates.refine(pen0, p_proj, st.upper_curvature, jit=True)
      d0 = pen0.effective_dual(ref0.x_hat)
      p_bar0 = tree.scale(1.0 - dampen, st.multiplier)
      al_feas0 = tree.norm(
          tree.scale(1.0 / st.penalty, tree.sub(d0, p_bar0)),
          kind=norm_kind,
      )
      raw_feas0 = tree.norm(bound_c.residual(ref0.x_hat), kind=norm_kind)
      feas0 = jnp.maximum(al_feas0, raw_feas0)
      opt0 = tree.norm(ref0.v_hat, kind=norm_kind)
      conv0 = jnp.logical_and(opt0 <= opt_tol_arr, feas0 <= feas_tol_arr)
      M0 = ref0.lipschitz_used
      return IALMTransformState(
          multiplier=st.multiplier,
          prev_feas_res=feas0,
          tracker=st.tracker.bootstrap(
              center=ref0.x_hat,
              center_obj=ref0.obj_val,
              dual=d0,
              subgrad=ref0.v_hat,
              opt_residual=opt0,
              feas_residual=feas0,
              converged=conv0,
          ),
          subproblem=PFAPDSubproblemCarry.create(
              ref0.x_hat,
              ref0.obj_val,
              lower_curvature=st.lower_curvature,
              upper_curvature=M0,
              never_increased_m=st.never_increased_m,
              decay_curvature=decay_curvature,
              beta=beta,
          ),
      )

    st_ready = jax.lax.cond(
        state.initialized,
        lambda s: s,
        _bootstrap,
        state,
    )

    def _when_converged(st: IALMTransformState) -> IALMTransformState:
      return dataclasses.replace(st, tracker=st.tracker.increment_step())

    def _when_active(st: IALMTransformState) -> IALMTransformState:
      pen_oracle = composite.PenalizedOracle(
          base=base_oracle,
          constraint=bound_c,
          multiplier=st.multiplier,
          penalty=st.penalty,
          dampen=dampen,
      )
      m_hat = st.lower_curvature
      pf_out, u_cand, v_cand, cand_opt = st.subproblem.step(
          pen_oracle,
          center=st.center,
          inner_iter=st.inner_iter,
          config=pf_cfg,
          opt_tol=opt_tol_arr,
          norm_kind=norm_kind,
      )
      cand_dual = pen_oracle.effective_dual(pf_out.y)
      p_bar = tree.scale(1.0 - dampen, st.multiplier)
      al_feas = tree.norm(
          tree.scale(1.0 / st.penalty, tree.sub(cand_dual, p_bar)),
          kind=norm_kind,
      )
      raw_feas = tree.norm(bound_c.residual(pf_out.y), kind=norm_kind)
      cand_feas = jnp.maximum(al_feas, raw_feas)
      upd_tr = st.tracker.update_incumbent(
          pf_out.y,
          cand_dual,
          v_cand,
          cand_opt,
          cand_feas,
          opt_tol_arr=opt_tol_arr,
          feas_tol_arr=feas_tol_arr,
      )
      conv_now = jnp.logical_and(
          upd_tr.opt_residual <= opt_tol_arr,
          upd_tr.feas_residual <= feas_tol_arr,
      )
      running_code = jnp.asarray(
          int(pf_acg.PFACGStatus.RUNNING), dtype=jnp.int32
      )
      still_running = jnp.logical_and(
          pf_out.status == running_code,
          jnp.logical_not(conv_now),
      )

      def _continue_inner() -> IALMTransformState:
        return IALMTransformState(
            multiplier=st.multiplier,
            prev_feas_res=st.prev_feas_res,
            tracker=dataclasses.replace(
                upd_tr,
                inner_iter=pf_out.iteration,
                total_steps=st.total_steps + 1,
            ),
            subproblem=st.subproblem.continue_from(pf_out),
        )

      def _finish_subproblem() -> IALMTransformState:
        phi_y = pen_oracle.value(pf_out.y)
        stage_opt_done = cand_opt <= opt_tol_arr
        rejected, dist_sq, M_from_acg = _evaluate_apd_subproblem(
            pf_out,
            center=st.center,
            center_obj=st.center_obj,
            phi_y=phi_y,
            u_cand=u_cand,
            v_cand=v_cand,
            m_hat=m_hat,
            theta=theta,
            eff_rho=eff_rho,
            done_flag=jnp.logical_or(conv_now, stage_opt_done),
        )

        def _reject_outer() -> IALMTransformState:
          grown_m = jnp.minimum(m_hat * alpha, m_max_arr)
          M_keep = jnp.maximum(st.upper_curvature, M_from_acg)
          return IALMTransformState(
              multiplier=st.multiplier,
              prev_feas_res=st.prev_feas_res,
              tracker=dataclasses.replace(
                  upd_tr,
                  stage_outer_iter=st.stage_outer_iter + 1,
                  outer_iter=st.outer_iter + 1,
                  inner_iter=jnp.asarray(0, dtype=jnp.int32),
                  total_steps=st.total_steps + 1,
              ),
              subproblem=PFAPDSubproblemCarry.create(
                  st.center,
                  st.center_obj,
                  lower_curvature=grown_m,
                  upper_curvature=M_keep,
                  never_increased_m=jnp.asarray(False, dtype=jnp.bool_),
                  decay_curvature=decay_curvature,
                  beta=beta,
              ),
          )

        def _accept_outer() -> IALMTransformState:
          m_next, M_est = _accepted_apd_curvatures(
              m_hat=m_hat,
              upper_curvature=st.upper_curvature,
              never_increased_m=st.never_increased_m,
              M_from_acg=M_from_acg,
              dist_sq=dist_sq,
              phi_y=phi_y,
              alpha=alpha,
              decay_curvature=decay_curvature,
              m_min_arr=m_min_arr,
          )
          next_stage_outer = st.stage_outer_iter + 1
          advance_alm = jnp.logical_and(
              jnp.logical_not(conv_now),
              jnp.logical_or(
                  stage_opt_done,
                  next_stage_outer >= max_apd_iters,
              ),
          )
          if dual_step == 1.0:
            cand_mult = cand_dual
          else:
            _, cand_mult = bound_c.penalty_and_dual(
                pf_out.y,
                multiplier=st.multiplier,
                penalty=dual_step * st.penalty,
                dampen=dampen,
            )
          next_mult = jax.tree.map(
              lambda cm, sm: jnp.where(advance_alm, cm, sm),
              cand_mult,
              st.multiplier,
          )
          grow_c = jnp.logical_and(
              advance_alm,
              cand_feas > jnp.maximum(
                  feas_tol_arr,
                  feas_decay * st.prev_feas_res,
              ),
          )
          c_grown = jnp.minimum(
              st.penalty * penalty_factor,
              jnp.asarray(penalty_max, dtype=dtype),
          )
          next_penalty = jnp.where(grow_c, c_grown, st.penalty)
          prev_feas_next = jnp.where(
              advance_alm,
              jnp.minimum(st.prev_feas_res, cand_feas),
              st.prev_feas_res,
          )
          M_next = jnp.where(
              grow_c,
              M_est * (next_penalty / st.penalty),
              M_est,
          )
          next_center = jax.tree.map(
              lambda bx, py: jnp.where(conv_now, bx, py),
              upd_tr.best_x,
              pf_out.y,
          )
          next_pen_oracle = composite.PenalizedOracle(
              base=base_oracle,
              constraint=bound_c,
              multiplier=next_mult,
              penalty=next_penalty,
              dampen=dampen,
          )
          next_center_obj = jnp.where(
              jnp.logical_or(advance_alm, conv_now),
              next_pen_oracle.value(next_center),
              phi_y,
          )
          return IALMTransformState(
              multiplier=next_mult,
              prev_feas_res=prev_feas_next,
              tracker=dataclasses.replace(
                  upd_tr,
                  converged=conv_now,
                  stage_iter=jnp.where(
                      advance_alm,
                      st.stage_iter + 1,
                      st.stage_iter,
                  ),
                  stage_outer_iter=jnp.where(
                      advance_alm,
                      jnp.asarray(0, dtype=jnp.int32),
                      next_stage_outer,
                  ),
                  outer_iter=st.outer_iter + 1,
                  inner_iter=jnp.asarray(0, dtype=jnp.int32),
                  total_steps=st.total_steps + 1,
                  center=next_center,
                  center_obj=next_center_obj,
                  penalty=next_penalty,
              ),
              subproblem=PFAPDSubproblemCarry.create(
                  next_center,
                  next_center_obj,
                  lower_curvature=m_next,
                  upper_curvature=M_next,
                  never_increased_m=st.never_increased_m,
                  decay_curvature=decay_curvature,
                  beta=beta,
              ),
          )

        return typing.cast(
            IALMTransformState,
            jax.lax.cond(rejected, _reject_outer, _accept_outer),
        )

      return typing.cast(
          IALMTransformState,
          jax.lax.cond(still_running, _continue_inner, _finish_subproblem),
      )

    next_state = typing.cast(
        IALMTransformState,
        jax.lax.cond(
            st_ready.converged,
            _when_converged,
            _when_active,
            st_ready,
        ),
    )
    target_primal = jax.tree.map(
        lambda bx, py: jnp.where(next_state.converged, bx, py),
        next_state.best_x,
        next_state.pf_y,
    )
    return tree.sub(target_primal, params), next_state

  return optax.GradientTransformationExtraArgs(init=init_fn, update=update_fn)
