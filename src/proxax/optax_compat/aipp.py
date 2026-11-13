"""Optax single-step `aipp` transformation."""

from __future__ import annotations

import dataclasses
import typing

import jax
import jax.numpy as jnp
import optax

from proxax import _typing
from proxax import certificates
from proxax import tree
from proxax.optax_compat._subproblems import ACGSubproblemCarry
from proxax.optax_compat._subproblems import ACGSubproblemMixin
from proxax.optax_compat._subproblems import OptaxTrackerMixin
from proxax.optax_compat._subproblems import _check_aipp_rejection
from proxax.optax_compat._subproblems import _initial_lam
from proxax.optax_compat._trackers import OptaxTracker
from proxax.oracle import composite
from proxax.oracle import terms
from proxax.prox import base as prox_base
from proxax.solvers.inner import acg

_ZERO_PROX: typing.Final = prox_base.Zero()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class AIPPTransformState(OptaxTrackerMixin, ACGSubproblemMixin):
  """Single-inner-step state machine for `proxax.optax_compat.aipp`."""

  tracker: OptaxTracker
  subproblem: ACGSubproblemCarry


def aipp(
    prox_op: prox_base.ProxOperator = _ZERO_PROX,
    *,
    opt_tol: float = 1e-5,
    norm_kind: str = "l2",
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
    max_inner_iters: int = 1000,
) -> optax.GradientTransformationExtraArgs:
  """Creates an Optax single-step `AIPP` transformation for `min f(x) + h(x)`.

  Each call to `tx.update(grads, state, params, value_fn=smooth_fn)` executes
  **one** inner `ACG` step on the active proximal subproblem and returns
  `updates` such that `optax.apply_updates(params, updates)` equals the feasible
  primal iterate `y_t in dom(h)` (and equals the certified stationary point
  `x_hat` once `state.converged` is True).

  Args:
    prox_op: Nonsmooth convex `ProxOperator` `h` (defaults to `Zero()`).
    opt_tol: Stationarity tolerance `||v_hat|| <= opt_tol`.
    norm_kind: Residual norm (`'l2'` or `'linf'`).
    sigma: Inner `ACG` relative proximal error parameter in `(0, 1)`.
    lower_curvature: Optional weak-convexity constant `m >= 0` of `value_fn`.
    upper_curvature: Optional global smoothness constant `L^+` of `value_fn`.
    lam_init: Optional initial proximal step size `lambda_0 > 0`.
    lam_min: Minimum allowed proximal step size.
    lam_max: Maximum allowed proximal step size.
    lam_increase: Growth factor for `lam` after fast inner convergence.
    lam_decrease: Shrink factor (`< 1`) when nonconvexity is detected.
    adaptive_prox: Whether to adapt `lam` (`None` = automatic).
    lipschitz_init: Initial smoothness estimate `L_0 > 0`.
    max_inner_iters: Maximum `ACG` steps per proximal subproblem.

  Returns:
    An `optax.GradientTransformationExtraArgs`.
  """
  has_known_m = lower_curvature is not None
  use_adaptive = (not has_known_m) if adaptive_prox is None else adaptive_prox
  acg_cfg = acg.ACGConfig(sigma=sigma, max_iters=max_inner_iters)

  def init_fn(params: _typing.PyTree) -> AIPPTransformState:
    p0 = prox_op.domain_project(tree.as_float_tree(params))
    dtype = tree.infer_dtype(p0)
    lam0 = _initial_lam(
        lam_init=lam_init,
        lower_curvature=lower_curvature,
        lipschitz_init=lipschitz_init,
        dtype=dtype,
    )
    return AIPPTransformState(
        tracker=OptaxTracker.uninitialized(p0),
        subproblem=ACGSubproblemCarry.create(
            p0,
            jnp.asarray(0.0, dtype=dtype),
            lam=lam0,
            lipschitz=jnp.asarray(lipschitz_init, dtype=dtype),
        ),
    )

  def update_fn(
      updates: _typing.PyTree,
      state: AIPPTransformState,
      params: _typing.PyTree | None = None,
      *,
      value: jax.Array | None = None,
      value_fn: _typing.SmoothFn | None = None,
      **extra_args: typing.Any,
  ) -> tuple[_typing.PyTree, AIPPTransformState]:
    del updates, value, extra_args
    if params is None:
      raise ValueError(
          "`proxax.optax_compat.aipp` requires `params` in `tx.update`."
      )
    if value_fn is None:
      raise ValueError(
          "`proxax.optax_compat.aipp` requires `value_fn=<smooth_objective>` "
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
    dtype = state.lipschitz.dtype
    tol_arr = jnp.asarray(opt_tol, dtype=dtype)

    def _bootstrap(st: AIPPTransformState) -> AIPPTransformState:
      p_proj = prox_op.domain_project(tree.as_float_tree(params))
      ref0 = certificates.refine(base_oracle, p_proj, st.lipschitz, jit=True)
      res0 = tree.norm(ref0.v_hat, kind=norm_kind)
      if lam_init is None and not has_known_m:
        lam0 = jnp.clip(
            2.0
            / jnp.maximum(
                ref0.lipschitz_used,
                jnp.asarray(1e-6, dtype=dtype),
            ),
            jnp.asarray(lam_min, dtype=dtype),
            jnp.asarray(lam_max, dtype=dtype),
        )
      else:
        lam0 = st.lam
      conv0 = res0 <= tol_arr
      return AIPPTransformState(
          tracker=st.tracker.bootstrap(
              center=ref0.x_hat,
              center_obj=ref0.obj_val,
              subgrad=ref0.v_hat,
              opt_residual=res0,
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

    def _when_converged(st: AIPPTransformState) -> AIPPTransformState:
      return dataclasses.replace(st, tracker=st.tracker.increment_step())

    def _when_active(st: AIPPTransformState) -> AIPPTransformState:
      acg_out, smooth_reg, inv_lam = st.subproblem.step(
          base_oracle,
          center=st.center,
          inner_iter=st.inner_iter,
          config=acg_cfg,
          has_known_m=has_known_m,
          lower_curvature=lower_curvature,
          use_adaptive=use_adaptive,
      )
      running_code = jnp.asarray(int(acg.ACGStatus.RUNNING), dtype=jnp.int32)
      still_running = acg_out.status == running_code

      def _continue_inner() -> AIPPTransformState:
        return AIPPTransformState(
            tracker=dataclasses.replace(
                st.tracker,
                inner_iter=acg_out.iteration,
                total_steps=st.total_steps + 1,
            ),
            subproblem=st.subproblem.continue_from(
                acg_out, smooth_reg=smooth_reg
            ),
        )

      def _finish_subproblem() -> AIPPTransformState:
        phi_y = base_oracle.value(acg_out.y)
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

        def _reject_outer() -> AIPPTransformState:
          shrunk = jnp.maximum(
              st.lam * lam_decrease,
              jnp.asarray(lam_min, dtype=dtype),
          )
          est_L = jnp.maximum(acg_out.lipschitz - smooth_reg, st.lipschitz)
          return AIPPTransformState(
              tracker=dataclasses.replace(
                  st.tracker,
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

        def _accept_outer() -> AIPPTransformState:
          est_L = jnp.maximum(
              acg_out.lipschitz - smooth_reg,
              jnp.asarray(1e-8, dtype=dtype),
          )
          ref = certificates.refine(base_oracle, acg_out.y, est_L, jit=True)
          cand_res = tree.norm(ref.v_hat, kind=norm_kind)
          upd_tr = st.tracker.update_incumbent(
              ref.x_hat,
              ref.v_hat,
              cand_res,
              check_finite=False,
          )
          conv = upd_tr.opt_residual <= tol_arr

          use_ref = ref.obj_val <= phi_y
          cand_center = jax.tree.map(
              lambda rx, yx: jnp.where(use_ref, rx, yx),
              ref.x_hat,
              acg_out.y,
          )
          cand_center_obj = jnp.where(use_ref, ref.obj_val, phi_y)
          next_center = jax.tree.map(
              lambda bx, cc: jnp.where(conv, bx, cc),
              upd_tr.best_x,
              cand_center,
          )
          next_center_obj = jnp.where(
              conv,
              base_oracle.value(upd_tr.best_x),
              cand_center_obj,
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
          return AIPPTransformState(
              tracker=dataclasses.replace(
                  upd_tr,
                  converged=conv,
                  outer_iter=st.outer_iter + 1,
                  inner_iter=jnp.asarray(0, dtype=jnp.int32),
                  total_steps=st.total_steps + 1,
                  center=next_center,
                  center_obj=next_center_obj,
              ),
              subproblem=ACGSubproblemCarry.create(
                  next_center,
                  next_center_obj,
                  lam=next_lam,
                  lipschitz=ref.lipschitz_used,
              ),
          )

        return typing.cast(
            AIPPTransformState,
            jax.lax.cond(rejected, _reject_outer, _accept_outer),
        )

      return typing.cast(
          AIPPTransformState,
          jax.lax.cond(still_running, _continue_inner, _finish_subproblem),
      )

    next_state = typing.cast(
        AIPPTransformState,
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

