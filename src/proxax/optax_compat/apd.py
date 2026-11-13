"""Optax single-step `apd` transformation."""

from __future__ import annotations

import dataclasses
import typing

import jax
import jax.numpy as jnp
import optax

from proxax import _typing
from proxax import certificates
from proxax import tree
from proxax.optax_compat._subproblems import OptaxTrackerMixin
from proxax.optax_compat._subproblems import PFAPDSubproblemCarry
from proxax.optax_compat._subproblems import PFAPDSubproblemMixin
from proxax.optax_compat._subproblems import _accepted_apd_curvatures
from proxax.optax_compat._subproblems import _evaluate_apd_subproblem
from proxax.optax_compat._subproblems import _initial_lower_curvature
from proxax.optax_compat._subproblems import _validate_apd_params
from proxax.optax_compat._trackers import OptaxTracker
from proxax.oracle import composite
from proxax.oracle import terms
from proxax.prox import base as prox_base
from proxax.solvers.inner import pf_acg

_ZERO_PROX: typing.Final = prox_base.Zero()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class APDTransformState(OptaxTrackerMixin, PFAPDSubproblemMixin):
  """Single-inner-step state machine for `proxax.optax_compat.apd`."""

  tracker: OptaxTracker
  subproblem: PFAPDSubproblemCarry


def apd(
    prox_op: prox_base.ProxOperator = _ZERO_PROX,
    *,
    opt_tol: float = 1e-5,
    norm_kind: str = "l2",
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
    max_inner_iters: int = 1000,
) -> optax.GradientTransformationExtraArgs:
  """Creates an Optax single-step `APD` transformation for `min f(x) + h(x)`.

  Each call to `tx.update(grads, state, params, value_fn=smooth_fn)` executes
  **one** inner `PF.ACG` step on the rescaled proximal subproblem and updates
  the refinement-free exact certificate `state.subgrad` at zero extra oracle
  cost.

  Args:
    prox_op: Nonsmooth convex `ProxOperator` `h` (defaults to `Zero()`).
    opt_tol: Stationarity tolerance `||v_hat|| <= opt_tol`.
    norm_kind: Residual norm (`'l2'` or `'linf'`).
    alpha: Multiplicative growth factor (`> 1`) for lower curvature `m_hat`.
    beta: Multiplicative growth factor (`> 1`) for inner smoothness `L`.
    theta: Proximal descent constant (`> 2`, default `4.0`).
    rho: Optional inner relative proximal error tolerance in `(0, 1)`.
    decay_curvature: Whether to apply practical curvature decay Eq. (6.1).
    lower_curvature: Optional weak-convexity constant `m >= 0` of `value_fn`.
    upper_curvature: Optional global smoothness constant `M` of `value_fn`.
    lower_curvature_init: Optional initial lower curvature `m_0 > 0`.
    lower_curvature_min: Floor `m_min > 0` guarding curvature decay.
    lower_curvature_max: Maximum allowed lower curvature estimate.
    lipschitz_init: Initial smoothness estimate `M_0 > 0`.
    max_inner_iters: Maximum `PF.ACG` steps per proximal subproblem.

  Returns:
    An `optax.GradientTransformationExtraArgs`.
  """
  eff_rho, pf_cfg = _validate_apd_params(
      alpha=alpha,
      beta=beta,
      theta=theta,
      rho=rho,
      decay_curvature=decay_curvature,
      max_inner_iters=max_inner_iters,
  )

  def init_fn(params: _typing.PyTree) -> APDTransformState:
    p0 = prox_op.domain_project(tree.as_float_tree(params))
    dtype = tree.infer_dtype(p0)
    m0 = _initial_lower_curvature(
        lower_curvature_init=lower_curvature_init,
        lower_curvature=lower_curvature,
        opt_tol=opt_tol,
        lower_curvature_min=lower_curvature_min,
        lower_curvature_max=lower_curvature_max,
        dtype=dtype,
    )
    M0 = jnp.asarray(lipschitz_init, dtype=dtype)
    return APDTransformState(
        tracker=OptaxTracker.uninitialized(p0),
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
      state: APDTransformState,
      params: _typing.PyTree | None = None,
      *,
      value: jax.Array | None = None,
      value_fn: _typing.SmoothFn | None = None,
      **extra_args: typing.Any,
  ) -> tuple[_typing.PyTree, APDTransformState]:
    del updates, value, extra_args
    if params is None:
      raise ValueError(
          "`proxax.optax_compat.apd` requires `params` in `tx.update`."
      )
    if value_fn is None:
      raise ValueError(
          "`proxax.optax_compat.apd` requires `value_fn=<smooth_objective>` "
          "to be passed as a keyword argument to `tx.update`."
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
    dtype = state.upper_curvature.dtype
    tol_arr = jnp.asarray(opt_tol, dtype=dtype)
    m_min_arr = jnp.asarray(lower_curvature_min, dtype=dtype)
    m_max_arr = jnp.asarray(lower_curvature_max, dtype=dtype)

    def _bootstrap(st: APDTransformState) -> APDTransformState:
      p_proj = prox_op.domain_project(tree.as_float_tree(params))
      ref0 = certificates.refine(
          base_oracle,
          p_proj,
          st.upper_curvature,
          jit=True,
      )
      res0 = tree.norm(ref0.v_hat, kind=norm_kind)
      conv0 = res0 <= tol_arr
      M0 = ref0.lipschitz_used
      return APDTransformState(
          tracker=st.tracker.bootstrap(
              center=ref0.x_hat,
              center_obj=ref0.obj_val,
              subgrad=ref0.v_hat,
              opt_residual=res0,
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

    def _when_converged(st: APDTransformState) -> APDTransformState:
      return dataclasses.replace(st, tracker=st.tracker.increment_step())

    def _when_active(st: APDTransformState) -> APDTransformState:
      m_hat = st.lower_curvature
      pf_out, u_cand, v_cand, cand_res = st.subproblem.step(
          base_oracle,
          center=st.center,
          inner_iter=st.inner_iter,
          config=pf_cfg,
          opt_tol=tol_arr,
          norm_kind=norm_kind,
      )
      upd_tr = st.tracker.update_incumbent(pf_out.y, v_cand, cand_res)
      conv_now = upd_tr.opt_residual <= tol_arr
      running_code = jnp.asarray(
          int(pf_acg.PFACGStatus.RUNNING), dtype=jnp.int32
      )
      still_running = jnp.logical_and(
          pf_out.status == running_code,
          jnp.logical_not(conv_now),
      )

      def _continue_inner() -> APDTransformState:
        return APDTransformState(
            tracker=dataclasses.replace(
                upd_tr,
                inner_iter=pf_out.iteration,
                total_steps=st.total_steps + 1,
            ),
            subproblem=st.subproblem.continue_from(pf_out),
        )

      def _finish_subproblem() -> APDTransformState:
        phi_y = base_oracle.value(pf_out.y)
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
            done_flag=conv_now,
        )

        def _reject_outer() -> APDTransformState:
          grown_m = jnp.minimum(m_hat * alpha, m_max_arr)
          M_keep = jnp.maximum(st.upper_curvature, M_from_acg)
          return APDTransformState(
              tracker=dataclasses.replace(
                  upd_tr,
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

        def _accept_outer() -> APDTransformState:
          m_next, M_next = _accepted_apd_curvatures(
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
          next_center = jax.tree.map(
              lambda bx, py: jnp.where(conv_now, bx, py),
              upd_tr.best_x,
              pf_out.y,
          )
          next_center_obj = jnp.where(
              conv_now,
              base_oracle.value(upd_tr.best_x),
              phi_y,
          )
          return APDTransformState(
              tracker=dataclasses.replace(
                  upd_tr,
                  converged=conv_now,
                  outer_iter=st.outer_iter + 1,
                  inner_iter=jnp.asarray(0, dtype=jnp.int32),
                  total_steps=st.total_steps + 1,
                  center=next_center,
                  center_obj=next_center_obj,
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
            APDTransformState,
            jax.lax.cond(rejected, _reject_outer, _accept_outer),
        )

      return typing.cast(
          APDTransformState,
          jax.lax.cond(still_running, _continue_inner, _finish_subproblem),
      )

    next_state = typing.cast(
        APDTransformState,
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

