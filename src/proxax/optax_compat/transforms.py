"""Optax gradient transformations for `proxax` operators and solvers."""

from __future__ import annotations

import dataclasses
import math
import typing
from collections import abc

import jax
import jax.numpy as jnp
import optax

from proxax import _typing
from proxax import certificates
from proxax import tree
from proxax.constraints import linear_op
from proxax.oracle import composite
from proxax.oracle import terms
from proxax.prox import base as prox_base
from proxax.solvers.inner import acg
from proxax.solvers.inner import pf_acg

StepSizeOrSchedule: typing.TypeAlias = (
    _typing.Scalar | abc.Callable[[jax.Array], _typing.Scalar]
)
_ZERO_PROX: typing.Final = prox_base.Zero()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ProximalState:
  """State for the `proximal` Optax transformation."""

  count: jax.Array


def proximal(
    prox_op: prox_base.ProxOperator,
    step_size: StepSizeOrSchedule = 1.0,
) -> optax.GradientTransformation:
  """Applies `prox_{step_size * h}(params + updates)` in an `optax.chain`.

  Designed to follow a gradient/momentum scaling step (e.g.
  `optax.chain(optax.scale(-lr), proxax.optax_compat.proximal(prox_op, lr))`)
  so `optax.apply_updates(params, updates)` lands inside `dom(h)`.

  Args:
    prox_op: Convex `ProxOperator` representing `h`.
    step_size: Positive scalar or `optax.Schedule` mapping `count -> tau_t`.

  Returns:
    An `optax.GradientTransformation`.
  """

  def init_fn(params: _typing.PyTree) -> ProximalState:
    del params
    return ProximalState(count=jnp.asarray(0, dtype=jnp.int32))

  def update_fn(
      updates: _typing.PyTree,
      state: ProximalState,
      params: _typing.PyTree | None = None,
  ) -> tuple[_typing.PyTree, ProximalState]:
    if params is None:
      raise ValueError(
          "`proxax.optax_compat.proximal` requires `params` to be passed to "
          "`tx.update(updates, state, params)`."
      )
    tau = step_size(state.count) if callable(step_size) else step_size
    candidate = tree.add(params, updates)
    projected = prox_op.prox(candidate, tau)
    new_updates = tree.sub(projected, params)
    next_count = optax.safe_int32_increment(state.count)
    return new_updates, ProximalState(count=next_count)

  return optax.GradientTransformation(init=init_fn, update=update_fn)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class AIPPTransformState:
  """Single-inner-step state machine for `proxax.optax_compat.aipp`."""

  initialized: jax.Array
  converged: jax.Array
  outer_iter: jax.Array
  inner_iter: jax.Array
  total_steps: jax.Array
  center: _typing.PyTree
  center_obj: jax.Array
  best_x: _typing.PyTree
  acg_x: _typing.PyTree
  acg_y: _typing.PyTree
  acg_u: _typing.PyTree
  acg_eta: jax.Array
  acg_a_sum: jax.Array
  sub_psi_x0: jax.Array
  lam: jax.Array
  lipschitz: jax.Array
  subgrad: _typing.PyTree
  opt_residual: jax.Array


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
    z0 = tree.zeros_like(p0)
    if lam_init is not None:
      lam0 = jnp.asarray(lam_init, dtype=dtype)
    elif lower_curvature is not None:
      lam0 = jnp.asarray(
          0.5 / max(lower_curvature, 1e-12)
          if lower_curvature > 0.0
          else 10.0 / max(lipschitz_init, 1e-6),
          dtype=dtype,
      )
    else:
      lam0 = jnp.asarray(2.0 / max(lipschitz_init, 1e-6), dtype=dtype)
    return AIPPTransformState(
        initialized=jnp.asarray(False, dtype=jnp.bool_),
        converged=jnp.asarray(False, dtype=jnp.bool_),
        outer_iter=jnp.asarray(0, dtype=jnp.int32),
        inner_iter=jnp.asarray(0, dtype=jnp.int32),
        total_steps=jnp.asarray(0, dtype=jnp.int32),
        center=p0,
        center_obj=jnp.asarray(0.0, dtype=dtype),
        best_x=p0,
        acg_x=p0,
        acg_y=p0,
        acg_u=z0,
        acg_eta=jnp.asarray(0.0, dtype=dtype),
        acg_a_sum=jnp.asarray(0.0, dtype=dtype),
        sub_psi_x0=jnp.asarray(0.0, dtype=dtype),
        lam=lam0,
        lipschitz=jnp.asarray(lipschitz_init, dtype=dtype),
        subgrad=z0,
        opt_residual=jnp.asarray(jnp.inf, dtype=dtype),
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
          initialized=jnp.asarray(True, dtype=jnp.bool_),
          converged=conv0,
          outer_iter=st.outer_iter,
          inner_iter=jnp.asarray(0, dtype=jnp.int32),
          total_steps=st.total_steps,
          center=ref0.x_hat,
          center_obj=ref0.obj_val,
          best_x=ref0.x_hat,
          acg_x=ref0.x_hat,
          acg_y=ref0.x_hat,
          acg_u=tree.zeros_like(ref0.x_hat),
          acg_eta=jnp.asarray(0.0, dtype=dtype),
          acg_a_sum=jnp.asarray(0.0, dtype=dtype),
          sub_psi_x0=ref0.obj_val,
          lam=lam0,
          lipschitz=ref0.lipschitz_used,
          subgrad=ref0.v_hat,
          opt_residual=res0,
      )

    st_ready = jax.lax.cond(
        state.initialized,
        lambda s: s,
        _bootstrap,
        state,
    )

    def _when_converged(st: AIPPTransformState) -> AIPPTransformState:
      return dataclasses.replace(st, total_steps=st.total_steps + 1)

    def _when_active(st: AIPPTransformState) -> AIPPTransformState:
      inv_lam = 1.0 / st.lam
      if has_known_m and not use_adaptive:
        m_arr = jnp.asarray(lower_curvature, dtype=dtype)
        smooth_reg = jnp.minimum(m_arr, inv_lam)
        prox_reg = jnp.maximum(jnp.asarray(0.0, dtype=dtype), inv_lam - m_arr)
      else:
        smooth_reg = 0.5 * inv_lam
        prox_reg = 0.5 * inv_lam

      sub_oracle = composite.ProxRegularizedOracle(
          base=base_oracle,
          center=st.center,
          smooth_reg=smooth_reg,
          prox_reg=prox_reg,
      )
      acg_in = acg.ACGState(
          x=st.acg_x,
          y=st.acg_y,
          x0=st.center,
          psi_x0=st.sub_psi_x0,
          psi_y=st.sub_psi_x0,
          u=st.acg_u,
          eta=st.acg_eta,
          a_sum=st.acg_a_sum,
          lipschitz=st.lipschitz + smooth_reg,
          iteration=st.inner_iter,
          status=jnp.asarray(int(acg.ACGStatus.RUNNING), dtype=jnp.int32),
          smooth_evals=jnp.asarray(0, dtype=jnp.int32),
          grad_evals=jnp.asarray(0, dtype=jnp.int32),
          prox_evals=jnp.asarray(0, dtype=jnp.int32),
      )
      acg_out = acg.step_acg(
          sub_oracle,
          acg_in,
          prox_center=st.center,
          lam=st.lam,
          config=acg_cfg,
          check_convexity=use_adaptive,
          jit=True,
      )
      running_code = jnp.asarray(int(acg.ACGStatus.RUNNING), dtype=jnp.int32)
      still_running = acg_out.status == running_code

      def _continue_inner() -> AIPPTransformState:
        est_L = jnp.maximum(
            acg_out.lipschitz - smooth_reg,
            0.5 * st.lipschitz,
        )
        return dataclasses.replace(
            st,
            inner_iter=acg_out.iteration,
            total_steps=st.total_steps + 1,
            acg_x=acg_out.x,
            acg_y=acg_out.y,
            acg_u=acg_out.u,
            acg_eta=acg_out.eta,
            acg_a_sum=acg_out.a_sum,
            lipschitz=est_L,
        )

      def _finish_subproblem() -> AIPPTransformState:
        phi_y = base_oracle.value(acg_out.y)
        rhs_vec = tree.add(
            tree.scale(inv_lam, tree.sub(st.center, acg_out.y)),
            acg_out.u,
        )
        req_desc = 0.5 * st.lam * (1.0 - sigma**2) * tree.norm_sq(rhs_vec)
        slack = 100.0 * tree.eps(dtype) * (1.0 + jnp.abs(st.center_obj))
        desc_ok = (st.center_obj - phi_y + slack) >= req_desc
        noncvx_code = jnp.asarray(
            int(acg.ACGStatus.NONCONVEX_DETECTED),
            dtype=jnp.int32,
        )
        rejected = jnp.logical_and(
            jnp.asarray(use_adaptive),
            jnp.logical_or(
                acg_out.status == noncvx_code,
                jnp.logical_not(desc_ok),
            ),
        )

        def _reject_outer() -> AIPPTransformState:
          shrunk = jnp.maximum(
              st.lam * lam_decrease,
              jnp.asarray(lam_min, dtype=dtype),
          )
          est_L = jnp.maximum(acg_out.lipschitz - smooth_reg, st.lipschitz)
          return dataclasses.replace(
              st,
              outer_iter=st.outer_iter + 1,
              inner_iter=jnp.asarray(0, dtype=jnp.int32),
              total_steps=st.total_steps + 1,
              acg_x=st.center,
              acg_y=st.center,
              acg_u=tree.zeros_like(st.center),
              acg_eta=jnp.asarray(0.0, dtype=dtype),
              acg_a_sum=jnp.asarray(0.0, dtype=dtype),
              lam=shrunk,
              lipschitz=est_L,
          )

        def _accept_outer() -> AIPPTransformState:
          est_L = jnp.maximum(
              acg_out.lipschitz - smooth_reg,
              jnp.asarray(1e-8, dtype=dtype),
          )
          ref = certificates.refine(base_oracle, acg_out.y, est_L, jit=True)
          cand_res = tree.norm(ref.v_hat, kind=norm_kind)
          is_better = cand_res <= st.opt_residual
          best_x = jax.tree.map(
              lambda cx, bx: jnp.where(is_better, cx, bx),
              ref.x_hat,
              st.best_x,
          )
          best_v = jax.tree.map(
              lambda cv, bv: jnp.where(is_better, cv, bv),
              ref.v_hat,
              st.subgrad,
          )
          best_res = jnp.where(is_better, cand_res, st.opt_residual)
          conv = best_res <= tol_arr

          use_ref = ref.obj_val <= phi_y
          cand_center = jax.tree.map(
              lambda rx, yx: jnp.where(use_ref, rx, yx),
              ref.x_hat,
              acg_out.y,
          )
          cand_center_obj = jnp.where(use_ref, ref.obj_val, phi_y)
          next_center = jax.tree.map(
              lambda bx, cc: jnp.where(conv, bx, cc),
              best_x,
              cand_center,
          )
          next_center_obj = jnp.where(
              conv,
              base_oracle.value(best_x),
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
              initialized=jnp.asarray(True, dtype=jnp.bool_),
              converged=conv,
              outer_iter=st.outer_iter + 1,
              inner_iter=jnp.asarray(0, dtype=jnp.int32),
              total_steps=st.total_steps + 1,
              center=next_center,
              center_obj=next_center_obj,
              best_x=best_x,
              acg_x=next_center,
              acg_y=next_center,
              acg_u=tree.zeros_like(next_center),
              acg_eta=jnp.asarray(0.0, dtype=dtype),
              acg_a_sum=jnp.asarray(0.0, dtype=dtype),
              sub_psi_x0=next_center_obj,
              lam=next_lam,
              lipschitz=ref.lipschitz_used,
              subgrad=best_v,
              opt_residual=best_res,
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


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class QPTransformState:
  """Single-inner-step state machine for `proxax.optax_compat.qp_aipp`."""

  initialized: jax.Array
  converged: jax.Array
  penalty_iter: jax.Array
  stage_outer_iter: jax.Array
  outer_iter: jax.Array
  inner_iter: jax.Array
  total_steps: jax.Array
  center: _typing.PyTree
  center_obj: jax.Array
  best_x: _typing.PyTree
  acg_x: _typing.PyTree
  acg_y: _typing.PyTree
  acg_u: _typing.PyTree
  acg_eta: jax.Array
  acg_a_sum: jax.Array
  sub_psi_x0: jax.Array
  penalty: jax.Array
  lam: jax.Array
  lipschitz: jax.Array
  dual: _typing.PyTree
  subgrad: _typing.PyTree
  opt_residual: jax.Array
  feas_residual: jax.Array


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
    constraints: `Constraint` or PyTree (e.g. `dict` / `tuple`) of `Constraint`s
      defining `A(x) in C`.
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
    z0 = tree.zeros_like(p0)
    bound_c = norm_c.bind(p0)
    dual0 = bound_c.zero_dual(p0)
    if lam_init is not None:
      lam0 = jnp.asarray(lam_init, dtype=dtype)
    elif lower_curvature is not None:
      lam0 = jnp.asarray(
          0.5 / max(lower_curvature, 1e-12)
          if lower_curvature > 0.0
          else 10.0 / max(lipschitz_init, 1e-6),
          dtype=dtype,
      )
    else:
      lam0 = jnp.asarray(2.0 / max(lipschitz_init, 1e-6), dtype=dtype)
    return QPTransformState(
        initialized=jnp.asarray(False, dtype=jnp.bool_),
        converged=jnp.asarray(False, dtype=jnp.bool_),
        penalty_iter=jnp.asarray(0, dtype=jnp.int32),
        stage_outer_iter=jnp.asarray(0, dtype=jnp.int32),
        outer_iter=jnp.asarray(0, dtype=jnp.int32),
        inner_iter=jnp.asarray(0, dtype=jnp.int32),
        total_steps=jnp.asarray(0, dtype=jnp.int32),
        center=p0,
        center_obj=jnp.asarray(0.0, dtype=dtype),
        best_x=p0,
        acg_x=p0,
        acg_y=p0,
        acg_u=z0,
        acg_eta=jnp.asarray(0.0, dtype=dtype),
        acg_a_sum=jnp.asarray(0.0, dtype=dtype),
        sub_psi_x0=jnp.asarray(0.0, dtype=dtype),
        penalty=jnp.asarray(penalty_init, dtype=dtype),
        lam=lam0,
        lipschitz=jnp.asarray(lipschitz_init, dtype=dtype),
        dual=dual0,
        subgrad=z0,
        opt_residual=jnp.asarray(jnp.inf, dtype=dtype),
        feas_residual=jnp.asarray(jnp.inf, dtype=dtype),
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
          initialized=jnp.asarray(True, dtype=jnp.bool_),
          converged=conv0,
          penalty_iter=st.penalty_iter,
          stage_outer_iter=jnp.asarray(0, dtype=jnp.int32),
          outer_iter=st.outer_iter,
          inner_iter=jnp.asarray(0, dtype=jnp.int32),
          total_steps=st.total_steps,
          center=ref0.x_hat,
          center_obj=ref0.obj_val,
          best_x=ref0.x_hat,
          acg_x=ref0.x_hat,
          acg_y=ref0.x_hat,
          acg_u=tree.zeros_like(ref0.x_hat),
          acg_eta=jnp.asarray(0.0, dtype=dtype),
          acg_a_sum=jnp.asarray(0.0, dtype=dtype),
          sub_psi_x0=ref0.obj_val,
          penalty=st.penalty,
          lam=lam0,
          lipschitz=ref0.lipschitz_used,
          dual=d0,
          subgrad=ref0.v_hat,
          opt_residual=opt0,
          feas_residual=feas0,
      )

    st_ready = jax.lax.cond(
        state.initialized,
        lambda s: s,
        _bootstrap,
        state,
    )

    def _when_converged(st: QPTransformState) -> QPTransformState:
      return dataclasses.replace(st, total_steps=st.total_steps + 1)

    def _when_active(st: QPTransformState) -> QPTransformState:
      pen_oracle = composite.PenalizedOracle(
          base=base_oracle,
          constraint=bound_c,
          multiplier=None,
          penalty=st.penalty,
          dampen=0.0,
      )
      inv_lam = 1.0 / st.lam
      if has_known_m and not use_adaptive:
        m_arr = jnp.asarray(lower_curvature, dtype=dtype)
        smooth_reg = jnp.minimum(m_arr, inv_lam)
        prox_reg = jnp.maximum(jnp.asarray(0.0, dtype=dtype), inv_lam - m_arr)
      else:
        smooth_reg = 0.5 * inv_lam
        prox_reg = 0.5 * inv_lam

      sub_oracle = composite.ProxRegularizedOracle(
          base=pen_oracle,
          center=st.center,
          smooth_reg=smooth_reg,
          prox_reg=prox_reg,
      )
      acg_in = acg.ACGState(
          x=st.acg_x,
          y=st.acg_y,
          x0=st.center,
          psi_x0=st.sub_psi_x0,
          psi_y=st.sub_psi_x0,
          u=st.acg_u,
          eta=st.acg_eta,
          a_sum=st.acg_a_sum,
          lipschitz=st.lipschitz + smooth_reg,
          iteration=st.inner_iter,
          status=jnp.asarray(int(acg.ACGStatus.RUNNING), dtype=jnp.int32),
          smooth_evals=jnp.asarray(0, dtype=jnp.int32),
          grad_evals=jnp.asarray(0, dtype=jnp.int32),
          prox_evals=jnp.asarray(0, dtype=jnp.int32),
      )
      acg_out = acg.step_acg(
          sub_oracle,
          acg_in,
          prox_center=st.center,
          lam=st.lam,
          config=acg_cfg,
          check_convexity=use_adaptive,
          jit=True,
      )
      running_code = jnp.asarray(int(acg.ACGStatus.RUNNING), dtype=jnp.int32)
      still_running = acg_out.status == running_code

      def _continue_inner() -> QPTransformState:
        est_L = jnp.maximum(
            acg_out.lipschitz - smooth_reg,
            0.5 * st.lipschitz,
        )
        return dataclasses.replace(
            st,
            inner_iter=acg_out.iteration,
            total_steps=st.total_steps + 1,
            acg_x=acg_out.x,
            acg_y=acg_out.y,
            acg_u=acg_out.u,
            acg_eta=acg_out.eta,
            acg_a_sum=acg_out.a_sum,
            lipschitz=est_L,
        )

      def _finish_subproblem() -> QPTransformState:
        phi_y = pen_oracle.value(acg_out.y)
        rhs_vec = tree.add(
            tree.scale(inv_lam, tree.sub(st.center, acg_out.y)),
            acg_out.u,
        )
        req_desc = 0.5 * st.lam * (1.0 - sigma**2) * tree.norm_sq(rhs_vec)
        slack = 100.0 * tree.eps(dtype) * (1.0 + jnp.abs(st.center_obj))
        desc_ok = (st.center_obj - phi_y + slack) >= req_desc
        noncvx_code = jnp.asarray(
            int(acg.ACGStatus.NONCONVEX_DETECTED),
            dtype=jnp.int32,
        )
        rejected = jnp.logical_and(
            jnp.asarray(use_adaptive),
            jnp.logical_or(
                acg_out.status == noncvx_code,
                jnp.logical_not(desc_ok),
            ),
        )

        def _reject_outer() -> QPTransformState:
          shrunk = jnp.maximum(
              st.lam * lam_decrease,
              jnp.asarray(lam_min, dtype=dtype),
          )
          est_L = jnp.maximum(acg_out.lipschitz - smooth_reg, st.lipschitz)
          return dataclasses.replace(
              st,
              stage_outer_iter=st.stage_outer_iter + 1,
              outer_iter=st.outer_iter + 1,
              inner_iter=jnp.asarray(0, dtype=jnp.int32),
              total_steps=st.total_steps + 1,
              acg_x=st.center,
              acg_y=st.center,
              acg_u=tree.zeros_like(st.center),
              acg_eta=jnp.asarray(0.0, dtype=dtype),
              acg_a_sum=jnp.asarray(0.0, dtype=dtype),
              lam=shrunk,
              lipschitz=est_L,
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
              initialized=jnp.asarray(True, dtype=jnp.bool_),
              converged=conv,
              penalty_iter=jnp.where(
                  advance_penalty,
                  st.penalty_iter + 1,
                  st.penalty_iter,
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
              acg_x=next_center,
              acg_y=next_center,
              acg_u=tree.zeros_like(next_center),
              acg_eta=jnp.asarray(0.0, dtype=dtype),
              acg_a_sum=jnp.asarray(0.0, dtype=dtype),
              sub_psi_x0=next_center_obj,
              penalty=next_penalty,
              lam=next_lam,
              lipschitz=next_L,
              dual=cand_dual,
              subgrad=ref.v_hat,
              opt_residual=cand_opt,
              feas_residual=cand_feas,
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
class APDTransformState:
  """Single-inner-step state machine for `proxax.optax_compat.apd`."""

  initialized: jax.Array
  converged: jax.Array
  outer_iter: jax.Array
  inner_iter: jax.Array
  total_steps: jax.Array
  center: _typing.PyTree
  center_obj: jax.Array
  best_x: _typing.PyTree
  pf_x: _typing.PyTree
  pf_y: _typing.PyTree
  pf_r: _typing.PyTree
  pf_a_sum: jax.Array
  pf_psi_y0: jax.Array
  pf_psi_y: jax.Array
  pf_lipschitz: jax.Array
  lower_curvature: jax.Array
  upper_curvature: jax.Array
  never_increased_m: jax.Array
  subgrad: _typing.PyTree
  opt_residual: jax.Array


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
  if alpha <= 1.0:
    raise ValueError("`alpha` must be strictly greater than 1.")
  if beta <= 1.0:
    raise ValueError("`beta` must be strictly greater than 1.")
  if theta <= 2.0:
    raise ValueError("`theta` must be strictly greater than 2.")
  if rho is not None and not (0.0 < rho < 1.0):
    raise ValueError("`rho` must lie in `(0, 1)`.")

  eff_rho = (
      rho
      if rho is not None
      else (
          1.0 / math.sqrt(alpha)
          if decay_curvature
          else 1.0 / math.sqrt(2.0 * alpha)
      )
  )
  pf_cfg = pf_acg.PFACGConfig(
      mu=0.5,
      sigma=eff_rho / math.sqrt(2.0),
      theta=theta,
      beta=beta,
      max_iters=max_inner_iters,
  )

  def _sub_lip_init(m_val: jax.Array, M_val: jax.Array) -> jax.Array:
    dtype = M_val.dtype
    if decay_curvature:
      M_trial = jnp.maximum(
          jnp.asarray(1e-8, dtype=dtype),
          M_val / (1.0 + 0.5 * beta),
      )
      return M_trial / (2.0 * m_val) + 1.0
    return jnp.maximum(
        jnp.asarray(1.0, dtype=dtype),
        M_val / (2.0 * m_val) + 1.0,
    )

  def init_fn(params: _typing.PyTree) -> APDTransformState:
    p0 = prox_op.domain_project(tree.as_float_tree(params))
    dtype = tree.infer_dtype(p0)
    z0 = tree.zeros_like(p0)
    if lower_curvature_init is not None:
      m0_f = lower_curvature_init
    elif lower_curvature is not None:
      m0_f = (
          lower_curvature
          if lower_curvature > 0.0
          else max(opt_tol, lower_curvature_min)
      )
    else:
      m0_f = 1.0
    m0 = jnp.clip(
        jnp.asarray(m0_f, dtype=dtype),
        jnp.asarray(lower_curvature_min, dtype=dtype),
        jnp.asarray(lower_curvature_max, dtype=dtype),
    )
    M0 = jnp.asarray(lipschitz_init, dtype=dtype)
    return APDTransformState(
        initialized=jnp.asarray(False, dtype=jnp.bool_),
        converged=jnp.asarray(False, dtype=jnp.bool_),
        outer_iter=jnp.asarray(0, dtype=jnp.int32),
        inner_iter=jnp.asarray(0, dtype=jnp.int32),
        total_steps=jnp.asarray(0, dtype=jnp.int32),
        center=p0,
        center_obj=jnp.asarray(0.0, dtype=dtype),
        best_x=p0,
        pf_x=p0,
        pf_y=p0,
        pf_r=z0,
        pf_a_sum=jnp.asarray(0.0, dtype=dtype),
        pf_psi_y0=jnp.asarray(0.0, dtype=dtype),
        pf_psi_y=jnp.asarray(0.0, dtype=dtype),
        pf_lipschitz=_sub_lip_init(m0, M0),
        lower_curvature=m0,
        upper_curvature=M0,
        never_increased_m=jnp.asarray(True, dtype=jnp.bool_),
        subgrad=z0,
        opt_residual=jnp.asarray(jnp.inf, dtype=dtype),
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
      psi0 = ref0.obj_val / (2.0 * st.lower_curvature)
      return APDTransformState(
          initialized=jnp.asarray(True, dtype=jnp.bool_),
          converged=conv0,
          outer_iter=st.outer_iter,
          inner_iter=jnp.asarray(0, dtype=jnp.int32),
          total_steps=st.total_steps,
          center=ref0.x_hat,
          center_obj=ref0.obj_val,
          best_x=ref0.x_hat,
          pf_x=ref0.x_hat,
          pf_y=ref0.x_hat,
          pf_r=tree.zeros_like(ref0.x_hat),
          pf_a_sum=jnp.asarray(0.0, dtype=dtype),
          pf_psi_y0=psi0,
          pf_psi_y=psi0,
          pf_lipschitz=_sub_lip_init(st.lower_curvature, M0),
          lower_curvature=st.lower_curvature,
          upper_curvature=M0,
          never_increased_m=st.never_increased_m,
          subgrad=ref0.v_hat,
          opt_residual=res0,
      )

    st_ready = jax.lax.cond(
        state.initialized,
        lambda s: s,
        _bootstrap,
        state,
    )

    def _when_converged(st: APDTransformState) -> APDTransformState:
      return dataclasses.replace(st, total_steps=st.total_steps + 1)

    def _when_active(st: APDTransformState) -> APDTransformState:
      m_hat = st.lower_curvature
      two_m = 2.0 * m_hat
      sub_oracle = composite.RescaledProxOracle(
          base=base_oracle,
          center=st.center,
          scale=1.0 / two_m,
      )
      running_code = jnp.asarray(
          int(pf_acg.PFACGStatus.RUNNING), dtype=jnp.int32
      )
      pf_in = pf_acg.PFACGState(
          x=st.pf_x,
          y=st.pf_y,
          y0=st.center,
          psi_y0=st.pf_psi_y0,
          psi_y=st.pf_psi_y,
          r=st.pf_r,
          a_sum=st.pf_a_sum,
          lipschitz=st.pf_lipschitz,
          iteration=st.inner_iter,
          status=running_code,
          smooth_evals=jnp.asarray(0, dtype=jnp.int32),
          grad_evals=jnp.asarray(0, dtype=jnp.int32),
          prox_evals=jnp.asarray(0, dtype=jnp.int32),
      )
      pf_out = pf_acg.step_pf_acg(
          sub_oracle,
          pf_in,
          config=pf_cfg,
          two_m=two_m,
          opt_tol=tol_arr,
          norm_kind=norm_kind,
          jit=True,
      )

      u_cand = tree.scale(two_m, pf_out.r)
      v_cand = tree.add(
          u_cand,
          tree.scale(two_m, tree.sub(st.center, pf_out.y)),
      )
      cand_res = tree.norm(v_cand, kind=norm_kind)
      cand_finite = jnp.logical_and(
          tree.all_finite(pf_out.y),
          jnp.isfinite(cand_res),
      )
      is_better = jnp.logical_and(cand_finite, cand_res <= st.opt_residual)
      best_x = jax.tree.map(
          lambda cy, bx: jnp.where(is_better, cy, bx),
          pf_out.y,
          st.best_x,
      )
      best_v = jax.tree.map(
          lambda cv, bv: jnp.where(is_better, cv, bv),
          v_cand,
          st.subgrad,
      )
      best_res = jnp.where(is_better, cand_res, st.opt_residual)
      conv_now = best_res <= tol_arr
      still_running = jnp.logical_and(
          pf_out.status == running_code,
          jnp.logical_not(conv_now),
      )

      def _continue_inner() -> APDTransformState:
        return dataclasses.replace(
            st,
            inner_iter=pf_out.iteration,
            total_steps=st.total_steps + 1,
            best_x=best_x,
            pf_x=pf_out.x,
            pf_y=pf_out.y,
            pf_r=pf_out.r,
            pf_a_sum=pf_out.a_sum,
            pf_psi_y=pf_out.psi_y,
            pf_lipschitz=pf_out.lipschitz,
            subgrad=best_v,
            opt_residual=best_res,
        )

      def _finish_subproblem() -> APDTransformState:
        phi_y = base_oracle.value(pf_out.y)
        eps_val = tree.eps(dtype)
        eps_sq = eps_val**2
        phi_slack = 100.0 * eps_val * (1.0 + jnp.abs(st.center_obj))
        desc_ok = (
            tree.norm_sq(v_cand)
            <= 2.0 * theta * m_hat * (st.center_obj - phi_y + phi_slack)
            + eps_sq
        )
        dist_sq = tree.norm_sq(tree.sub(pf_out.y, st.center))
        prox_err_ok = (
            tree.norm_sq(u_cand)
            <= 2.0 * ((eff_rho * m_hat) ** 2) * dist_sq + eps_sq
        )
        bad_code = jnp.asarray(
            int(pf_acg.PFACGStatus.BAD_TERMINATION),
            dtype=jnp.int32,
        )
        ls_fail_code = jnp.asarray(
            int(pf_acg.PFACGStatus.LINE_SEARCH_FAILED),
            dtype=jnp.int32,
        )
        rejected = jnp.logical_and(
            jnp.logical_not(conv_now),
            jnp.logical_and(
                pf_out.status != ls_fail_code,
                jnp.logical_or(
                    pf_out.status == bad_code,
                    jnp.logical_not(jnp.logical_and(desc_ok, prox_err_ok)),
                ),
            ),
        )
        M_from_acg = jnp.maximum(
            jnp.asarray(1e-8, dtype=dtype),
            two_m * (pf_out.lipschitz - 1.0),
        )

        def _reject_outer() -> APDTransformState:
          grown_m = jnp.minimum(m_hat * alpha, m_max_arr)
          M_keep = jnp.maximum(st.upper_curvature, M_from_acg)
          psi_y0_next = st.center_obj / (2.0 * grown_m)
          return dataclasses.replace(
              st,
              outer_iter=st.outer_iter + 1,
              inner_iter=jnp.asarray(0, dtype=jnp.int32),
              total_steps=st.total_steps + 1,
              best_x=best_x,
              pf_x=st.center,
              pf_y=st.center,
              pf_r=tree.zeros_like(st.center),
              pf_a_sum=jnp.asarray(0.0, dtype=dtype),
              pf_psi_y0=psi_y0_next,
              pf_psi_y=psi_y0_next,
              pf_lipschitz=_sub_lip_init(grown_m, M_keep),
              lower_curvature=grown_m,
              upper_curvature=M_keep,
              never_increased_m=jnp.asarray(False, dtype=jnp.bool_),
              subgrad=best_v,
              opt_residual=best_res,
          )

        def _accept_outer() -> APDTransformState:
          quad_curv = 0.5 * M_from_acg * dist_sq
          noise_floor = 100.0 * eps_val * (1.0 + jnp.abs(phi_y))
          M_next = jnp.where(
              quad_curv > noise_floor,
              M_from_acg,
              jnp.maximum(M_from_acg, st.upper_curvature),
          )
          if decay_curvature:
            m_relaxed = jnp.maximum(m_min_arr, m_hat / (1.0 + 0.5 * alpha))
            m_next = jnp.where(dist_sq > eps_sq, m_relaxed, m_hat)
          else:
            m_bitonic = jnp.maximum(m_min_arr, m_hat / alpha)
            m_next = jnp.where(st.never_increased_m, m_bitonic, m_hat)

          next_center = jax.tree.map(
              lambda bx, py: jnp.where(conv_now, bx, py),
              best_x,
              pf_out.y,
          )
          next_center_obj = jnp.where(
              conv_now,
              base_oracle.value(best_x),
              phi_y,
          )
          psi_y0_next = next_center_obj / (2.0 * m_next)
          return APDTransformState(
              initialized=jnp.asarray(True, dtype=jnp.bool_),
              converged=conv_now,
              outer_iter=st.outer_iter + 1,
              inner_iter=jnp.asarray(0, dtype=jnp.int32),
              total_steps=st.total_steps + 1,
              center=next_center,
              center_obj=next_center_obj,
              best_x=best_x,
              pf_x=next_center,
              pf_y=next_center,
              pf_r=tree.zeros_like(next_center),
              pf_a_sum=jnp.asarray(0.0, dtype=dtype),
              pf_psi_y0=psi_y0_next,
              pf_psi_y=psi_y0_next,
              pf_lipschitz=_sub_lip_init(m_next, M_next),
              lower_curvature=m_next,
              upper_curvature=M_next,
              never_increased_m=st.never_increased_m,
              subgrad=best_v,
              opt_residual=best_res,
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


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class QPAPDTransformState:
  """Single-inner-step state machine for `proxax.optax_compat.qp_apd`."""

  initialized: jax.Array
  converged: jax.Array
  penalty_iter: jax.Array
  stage_outer_iter: jax.Array
  outer_iter: jax.Array
  inner_iter: jax.Array
  total_steps: jax.Array
  center: _typing.PyTree
  center_obj: jax.Array
  best_x: _typing.PyTree
  pf_x: _typing.PyTree
  pf_y: _typing.PyTree
  pf_r: _typing.PyTree
  pf_a_sum: jax.Array
  pf_psi_y0: jax.Array
  pf_psi_y: jax.Array
  pf_lipschitz: jax.Array
  penalty: jax.Array
  lower_curvature: jax.Array
  upper_curvature: jax.Array
  never_increased_m: jax.Array
  dual: _typing.PyTree
  subgrad: _typing.PyTree
  opt_residual: jax.Array
  feas_residual: jax.Array


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
  if alpha <= 1.0:
    raise ValueError("`alpha` must be strictly greater than 1.")
  if beta <= 1.0:
    raise ValueError("`beta` must be strictly greater than 1.")
  if theta <= 2.0:
    raise ValueError("`theta` must be strictly greater than 2.")
  if rho is not None and not (0.0 < rho < 1.0):
    raise ValueError("`rho` must lie in `(0, 1)`.")

  eff_rho = (
      rho
      if rho is not None
      else (
          1.0 / math.sqrt(alpha)
          if decay_curvature
          else 1.0 / math.sqrt(2.0 * alpha)
      )
  )
  pf_cfg = pf_acg.PFACGConfig(
      mu=0.5,
      sigma=eff_rho / math.sqrt(2.0),
      theta=theta,
      beta=beta,
      max_iters=max_inner_iters,
  )

  def _sub_lip_init(m_val: jax.Array, M_val: jax.Array) -> jax.Array:
    dtype = M_val.dtype
    if decay_curvature:
      M_trial = jnp.maximum(
          jnp.asarray(1e-8, dtype=dtype),
          M_val / (1.0 + 0.5 * beta),
      )
      return M_trial / (2.0 * m_val) + 1.0
    return jnp.maximum(
        jnp.asarray(1.0, dtype=dtype),
        M_val / (2.0 * m_val) + 1.0,
    )

  def init_fn(params: _typing.PyTree) -> QPAPDTransformState:
    p0 = prox_op.domain_project(tree.as_float_tree(params))
    dtype = tree.infer_dtype(p0)
    z0 = tree.zeros_like(p0)
    bound_c = norm_c.bind(p0)
    dual0 = bound_c.zero_dual(p0)
    if lower_curvature_init is not None:
      m0_f = lower_curvature_init
    elif lower_curvature is not None:
      m0_f = (
          lower_curvature
          if lower_curvature > 0.0
          else max(opt_tol, lower_curvature_min)
      )
    else:
      m0_f = 1.0
    m0 = jnp.clip(
        jnp.asarray(m0_f, dtype=dtype),
        jnp.asarray(lower_curvature_min, dtype=dtype),
        jnp.asarray(lower_curvature_max, dtype=dtype),
    )
    M0 = jnp.asarray(lipschitz_init, dtype=dtype)
    return QPAPDTransformState(
        initialized=jnp.asarray(False, dtype=jnp.bool_),
        converged=jnp.asarray(False, dtype=jnp.bool_),
        penalty_iter=jnp.asarray(0, dtype=jnp.int32),
        stage_outer_iter=jnp.asarray(0, dtype=jnp.int32),
        outer_iter=jnp.asarray(0, dtype=jnp.int32),
        inner_iter=jnp.asarray(0, dtype=jnp.int32),
        total_steps=jnp.asarray(0, dtype=jnp.int32),
        center=p0,
        center_obj=jnp.asarray(0.0, dtype=dtype),
        best_x=p0,
        pf_x=p0,
        pf_y=p0,
        pf_r=z0,
        pf_a_sum=jnp.asarray(0.0, dtype=dtype),
        pf_psi_y0=jnp.asarray(0.0, dtype=dtype),
        pf_psi_y=jnp.asarray(0.0, dtype=dtype),
        pf_lipschitz=_sub_lip_init(m0, M0),
        penalty=jnp.asarray(penalty_init, dtype=dtype),
        lower_curvature=m0,
        upper_curvature=M0,
        never_increased_m=jnp.asarray(True, dtype=jnp.bool_),
        dual=dual0,
        subgrad=z0,
        opt_residual=jnp.asarray(jnp.inf, dtype=dtype),
        feas_residual=jnp.asarray(jnp.inf, dtype=dtype),
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
    tiny = jnp.asarray(1e-30, dtype=dtype)

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
      psi0 = ref0.obj_val / (2.0 * st.lower_curvature)
      return QPAPDTransformState(
          initialized=jnp.asarray(True, dtype=jnp.bool_),
          converged=conv0,
          penalty_iter=st.penalty_iter,
          stage_outer_iter=jnp.asarray(0, dtype=jnp.int32),
          outer_iter=st.outer_iter,
          inner_iter=jnp.asarray(0, dtype=jnp.int32),
          total_steps=st.total_steps,
          center=ref0.x_hat,
          center_obj=ref0.obj_val,
          best_x=ref0.x_hat,
          pf_x=ref0.x_hat,
          pf_y=ref0.x_hat,
          pf_r=tree.zeros_like(ref0.x_hat),
          pf_a_sum=jnp.asarray(0.0, dtype=dtype),
          pf_psi_y0=psi0,
          pf_psi_y=psi0,
          pf_lipschitz=_sub_lip_init(st.lower_curvature, M0),
          penalty=st.penalty,
          lower_curvature=st.lower_curvature,
          upper_curvature=M0,
          never_increased_m=st.never_increased_m,
          dual=d0,
          subgrad=ref0.v_hat,
          opt_residual=opt0,
          feas_residual=feas0,
      )

    st_ready = jax.lax.cond(
        state.initialized,
        lambda s: s,
        _bootstrap,
        state,
    )

    def _when_converged(st: QPAPDTransformState) -> QPAPDTransformState:
      return dataclasses.replace(st, total_steps=st.total_steps + 1)

    def _when_active(st: QPAPDTransformState) -> QPAPDTransformState:
      pen_oracle = composite.PenalizedOracle(
          base=base_oracle,
          constraint=bound_c,
          multiplier=None,
          penalty=st.penalty,
          dampen=0.0,
      )
      m_hat = st.lower_curvature
      two_m = 2.0 * m_hat
      sub_oracle = composite.RescaledProxOracle(
          base=pen_oracle,
          center=st.center,
          scale=1.0 / two_m,
      )
      running_code = jnp.asarray(
          int(pf_acg.PFACGStatus.RUNNING), dtype=jnp.int32
      )
      pf_in = pf_acg.PFACGState(
          x=st.pf_x,
          y=st.pf_y,
          y0=st.center,
          psi_y0=st.pf_psi_y0,
          psi_y=st.pf_psi_y,
          r=st.pf_r,
          a_sum=st.pf_a_sum,
          lipschitz=st.pf_lipschitz,
          iteration=st.inner_iter,
          status=running_code,
          smooth_evals=jnp.asarray(0, dtype=jnp.int32),
          grad_evals=jnp.asarray(0, dtype=jnp.int32),
          prox_evals=jnp.asarray(0, dtype=jnp.int32),
      )
      pf_out = pf_acg.step_pf_acg(
          sub_oracle,
          pf_in,
          config=pf_cfg,
          two_m=two_m,
          opt_tol=opt_tol_arr,
          norm_kind=norm_kind,
          jit=True,
      )

      u_cand = tree.scale(two_m, pf_out.r)
      v_cand = tree.add(
          u_cand,
          tree.scale(two_m, tree.sub(st.center, pf_out.y)),
      )
      cand_opt = tree.norm(v_cand, kind=norm_kind)
      cand_feas = tree.norm(bound_c.residual(pf_out.y), kind=norm_kind)
      cand_dual = pen_oracle.effective_dual(pf_out.y)
      cand_finite = jnp.logical_and(
          tree.all_finite(pf_out.y),
          jnp.logical_and(jnp.isfinite(cand_opt), jnp.isfinite(cand_feas)),
      )
      cand_score = jnp.maximum(
          cand_opt / jnp.maximum(opt_tol_arr, tiny),
          cand_feas / jnp.maximum(feas_tol_arr, tiny),
      )
      prev_score = jnp.maximum(
          st.opt_residual / jnp.maximum(opt_tol_arr, tiny),
          st.feas_residual / jnp.maximum(feas_tol_arr, tiny),
      )
      is_better = jnp.logical_and(cand_finite, cand_score <= prev_score)
      best_x = jax.tree.map(
          lambda cy, bx: jnp.where(is_better, cy, bx),
          pf_out.y,
          st.best_x,
      )
      best_dual = jax.tree.map(
          lambda cd, bd: jnp.where(is_better, cd, bd),
          cand_dual,
          st.dual,
      )
      best_v = jax.tree.map(
          lambda cv, bv: jnp.where(is_better, cv, bv),
          v_cand,
          st.subgrad,
      )
      best_opt = jnp.where(is_better, cand_opt, st.opt_residual)
      best_feas = jnp.where(is_better, cand_feas, st.feas_residual)
      conv_now = jnp.logical_and(
          best_opt <= opt_tol_arr,
          best_feas <= feas_tol_arr,
      )
      still_running = jnp.logical_and(
          pf_out.status == running_code,
          jnp.logical_not(conv_now),
      )

      def _continue_inner() -> QPAPDTransformState:
        return dataclasses.replace(
            st,
            inner_iter=pf_out.iteration,
            total_steps=st.total_steps + 1,
            best_x=best_x,
            pf_x=pf_out.x,
            pf_y=pf_out.y,
            pf_r=pf_out.r,
            pf_a_sum=pf_out.a_sum,
            pf_psi_y=pf_out.psi_y,
            pf_lipschitz=pf_out.lipschitz,
            dual=best_dual,
            subgrad=best_v,
            opt_residual=best_opt,
            feas_residual=best_feas,
        )

      def _finish_subproblem() -> QPAPDTransformState:
        phi_y = pen_oracle.value(pf_out.y)
        eps_val = tree.eps(dtype)
        eps_sq = eps_val**2
        phi_slack = 100.0 * eps_val * (1.0 + jnp.abs(st.center_obj))
        desc_ok = (
            tree.norm_sq(v_cand)
            <= 2.0 * theta * m_hat * (st.center_obj - phi_y + phi_slack)
            + eps_sq
        )
        dist_sq = tree.norm_sq(tree.sub(pf_out.y, st.center))
        prox_err_ok = (
            tree.norm_sq(u_cand)
            <= 2.0 * ((eff_rho * m_hat) ** 2) * dist_sq + eps_sq
        )
        bad_code = jnp.asarray(
            int(pf_acg.PFACGStatus.BAD_TERMINATION),
            dtype=jnp.int32,
        )
        ls_fail_code = jnp.asarray(
            int(pf_acg.PFACGStatus.LINE_SEARCH_FAILED),
            dtype=jnp.int32,
        )
        stage_opt_done = cand_opt <= opt_tol_arr
        rejected = jnp.logical_and(
            jnp.logical_not(jnp.logical_or(conv_now, stage_opt_done)),
            jnp.logical_and(
                pf_out.status != ls_fail_code,
                jnp.logical_or(
                    pf_out.status == bad_code,
                    jnp.logical_not(jnp.logical_and(desc_ok, prox_err_ok)),
                ),
            ),
        )
        M_from_acg = jnp.maximum(
            jnp.asarray(1e-8, dtype=dtype),
            two_m * (pf_out.lipschitz - 1.0),
        )

        def _reject_outer() -> QPAPDTransformState:
          grown_m = jnp.minimum(m_hat * alpha, m_max_arr)
          M_keep = jnp.maximum(st.upper_curvature, M_from_acg)
          psi_y0_next = st.center_obj / (2.0 * grown_m)
          return dataclasses.replace(
              st,
              stage_outer_iter=st.stage_outer_iter + 1,
              outer_iter=st.outer_iter + 1,
              inner_iter=jnp.asarray(0, dtype=jnp.int32),
              total_steps=st.total_steps + 1,
              best_x=best_x,
              pf_x=st.center,
              pf_y=st.center,
              pf_r=tree.zeros_like(st.center),
              pf_a_sum=jnp.asarray(0.0, dtype=dtype),
              pf_psi_y0=psi_y0_next,
              pf_psi_y=psi_y0_next,
              pf_lipschitz=_sub_lip_init(grown_m, M_keep),
              lower_curvature=grown_m,
              upper_curvature=M_keep,
              never_increased_m=jnp.asarray(False, dtype=jnp.bool_),
              dual=best_dual,
              subgrad=best_v,
              opt_residual=best_opt,
              feas_residual=best_feas,
          )

        def _accept_outer() -> QPAPDTransformState:
          quad_curv = 0.5 * M_from_acg * dist_sq
          noise_floor = 100.0 * eps_val * (1.0 + jnp.abs(phi_y))
          M_est = jnp.where(
              quad_curv > noise_floor,
              M_from_acg,
              jnp.maximum(M_from_acg, st.upper_curvature),
          )
          if decay_curvature:
            m_relaxed = jnp.maximum(m_min_arr, m_hat / (1.0 + 0.5 * alpha))
            m_next = jnp.where(dist_sq > eps_sq, m_relaxed, m_hat)
          else:
            m_bitonic = jnp.maximum(m_min_arr, m_hat / alpha)
            m_next = jnp.where(st.never_increased_m, m_bitonic, m_hat)

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
              best_x,
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
          psi_y0_next = next_center_obj / (2.0 * m_next)
          return QPAPDTransformState(
              initialized=jnp.asarray(True, dtype=jnp.bool_),
              converged=conv_now,
              penalty_iter=jnp.where(
                  advance_penalty,
                  st.penalty_iter + 1,
                  st.penalty_iter,
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
              best_x=best_x,
              pf_x=next_center,
              pf_y=next_center,
              pf_r=tree.zeros_like(next_center),
              pf_a_sum=jnp.asarray(0.0, dtype=dtype),
              pf_psi_y0=psi_y0_next,
              pf_psi_y=psi_y0_next,
              pf_lipschitz=_sub_lip_init(m_next, M_next),
              penalty=next_penalty,
              lower_curvature=m_next,
              upper_curvature=M_next,
              never_increased_m=st.never_increased_m,
              dual=best_dual,
              subgrad=best_v,
              opt_residual=best_opt,
              feas_residual=best_feas,
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
class IALMTransformState:
  """Single-inner-step state machine for `proxax.optax_compat.ialm`."""

  initialized: jax.Array
  converged: jax.Array
  alm_iter: jax.Array
  stage_outer_iter: jax.Array
  outer_iter: jax.Array
  inner_iter: jax.Array
  total_steps: jax.Array
  center: _typing.PyTree
  center_obj: jax.Array
  best_x: _typing.PyTree
  pf_x: _typing.PyTree
  pf_y: _typing.PyTree
  pf_r: _typing.PyTree
  pf_a_sum: jax.Array
  pf_psi_y0: jax.Array
  pf_psi_y: jax.Array
  pf_lipschitz: jax.Array
  multiplier: _typing.PyTree
  penalty: jax.Array
  prev_feas_res: jax.Array
  lower_curvature: jax.Array
  upper_curvature: jax.Array
  never_increased_m: jax.Array
  dual: _typing.PyTree
  subgrad: _typing.PyTree
  opt_residual: jax.Array
  feas_residual: jax.Array


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
  if alpha <= 1.0:
    raise ValueError("`alpha` must be strictly greater than 1.")
  if beta <= 1.0:
    raise ValueError("`beta` must be strictly greater than 1.")
  if theta <= 2.0:
    raise ValueError("`theta` must be strictly greater than 2.")
  if rho is not None and not (0.0 < rho < 1.0):
    raise ValueError("`rho` must lie in `(0, 1)`.")

  eff_rho = (
      rho
      if rho is not None
      else (
          1.0 / math.sqrt(alpha)
          if decay_curvature
          else 1.0 / math.sqrt(2.0 * alpha)
      )
  )
  pf_cfg = pf_acg.PFACGConfig(
      mu=0.5,
      sigma=eff_rho / math.sqrt(2.0),
      theta=theta,
      beta=beta,
      max_iters=max_inner_iters,
  )

  def _sub_lip_init(m_val: jax.Array, M_val: jax.Array) -> jax.Array:
    dtype = M_val.dtype
    if decay_curvature:
      M_trial = jnp.maximum(
          jnp.asarray(1e-8, dtype=dtype),
          M_val / (1.0 + 0.5 * beta),
      )
      return M_trial / (2.0 * m_val) + 1.0
    return jnp.maximum(
        jnp.asarray(1.0, dtype=dtype),
        M_val / (2.0 * m_val) + 1.0,
    )

  def init_fn(params: _typing.PyTree) -> IALMTransformState:
    p0 = prox_op.domain_project(tree.as_float_tree(params))
    dtype = tree.infer_dtype(p0)
    z0 = tree.zeros_like(p0)
    bound_c = norm_c.bind(p0)
    dual0 = bound_c.zero_dual(p0)
    if lower_curvature_init is not None:
      m0_f = lower_curvature_init
    elif lower_curvature is not None:
      m0_f = (
          lower_curvature
          if lower_curvature > 0.0
          else max(opt_tol, lower_curvature_min)
      )
    else:
      m0_f = 1.0
    m0 = jnp.clip(
        jnp.asarray(m0_f, dtype=dtype),
        jnp.asarray(lower_curvature_min, dtype=dtype),
        jnp.asarray(lower_curvature_max, dtype=dtype),
    )
    M0 = jnp.asarray(lipschitz_init, dtype=dtype)
    return IALMTransformState(
        initialized=jnp.asarray(False, dtype=jnp.bool_),
        converged=jnp.asarray(False, dtype=jnp.bool_),
        alm_iter=jnp.asarray(0, dtype=jnp.int32),
        stage_outer_iter=jnp.asarray(0, dtype=jnp.int32),
        outer_iter=jnp.asarray(0, dtype=jnp.int32),
        inner_iter=jnp.asarray(0, dtype=jnp.int32),
        total_steps=jnp.asarray(0, dtype=jnp.int32),
        center=p0,
        center_obj=jnp.asarray(0.0, dtype=dtype),
        best_x=p0,
        pf_x=p0,
        pf_y=p0,
        pf_r=z0,
        pf_a_sum=jnp.asarray(0.0, dtype=dtype),
        pf_psi_y0=jnp.asarray(0.0, dtype=dtype),
        pf_psi_y=jnp.asarray(0.0, dtype=dtype),
        pf_lipschitz=_sub_lip_init(m0, M0),
        multiplier=dual0,
        penalty=jnp.asarray(penalty_init, dtype=dtype),
        prev_feas_res=jnp.asarray(jnp.inf, dtype=dtype),
        lower_curvature=m0,
        upper_curvature=M0,
        never_increased_m=jnp.asarray(True, dtype=jnp.bool_),
        dual=dual0,
        subgrad=z0,
        opt_residual=jnp.asarray(jnp.inf, dtype=dtype),
        feas_residual=jnp.asarray(jnp.inf, dtype=dtype),
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
    tiny = jnp.asarray(1e-30, dtype=dtype)

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
      psi0 = ref0.obj_val / (2.0 * st.lower_curvature)
      return IALMTransformState(
          initialized=jnp.asarray(True, dtype=jnp.bool_),
          converged=conv0,
          alm_iter=st.alm_iter,
          stage_outer_iter=jnp.asarray(0, dtype=jnp.int32),
          outer_iter=st.outer_iter,
          inner_iter=jnp.asarray(0, dtype=jnp.int32),
          total_steps=st.total_steps,
          center=ref0.x_hat,
          center_obj=ref0.obj_val,
          best_x=ref0.x_hat,
          pf_x=ref0.x_hat,
          pf_y=ref0.x_hat,
          pf_r=tree.zeros_like(ref0.x_hat),
          pf_a_sum=jnp.asarray(0.0, dtype=dtype),
          pf_psi_y0=psi0,
          pf_psi_y=psi0,
          pf_lipschitz=_sub_lip_init(st.lower_curvature, M0),
          multiplier=st.multiplier,
          penalty=st.penalty,
          prev_feas_res=feas0,
          lower_curvature=st.lower_curvature,
          upper_curvature=M0,
          never_increased_m=st.never_increased_m,
          dual=d0,
          subgrad=ref0.v_hat,
          opt_residual=opt0,
          feas_residual=feas0,
      )

    st_ready = jax.lax.cond(
        state.initialized,
        lambda s: s,
        _bootstrap,
        state,
    )

    def _when_converged(st: IALMTransformState) -> IALMTransformState:
      return dataclasses.replace(st, total_steps=st.total_steps + 1)

    def _when_active(st: IALMTransformState) -> IALMTransformState:
      pen_oracle = composite.PenalizedOracle(
          base=base_oracle,
          constraint=bound_c,
          multiplier=st.multiplier,
          penalty=st.penalty,
          dampen=dampen,
      )
      m_hat = st.lower_curvature
      two_m = 2.0 * m_hat
      sub_oracle = composite.RescaledProxOracle(
          base=pen_oracle,
          center=st.center,
          scale=1.0 / two_m,
      )
      running_code = jnp.asarray(
          int(pf_acg.PFACGStatus.RUNNING), dtype=jnp.int32
      )
      pf_in = pf_acg.PFACGState(
          x=st.pf_x,
          y=st.pf_y,
          y0=st.center,
          psi_y0=st.pf_psi_y0,
          psi_y=st.pf_psi_y,
          r=st.pf_r,
          a_sum=st.pf_a_sum,
          lipschitz=st.pf_lipschitz,
          iteration=st.inner_iter,
          status=running_code,
          smooth_evals=jnp.asarray(0, dtype=jnp.int32),
          grad_evals=jnp.asarray(0, dtype=jnp.int32),
          prox_evals=jnp.asarray(0, dtype=jnp.int32),
      )
      pf_out = pf_acg.step_pf_acg(
          sub_oracle,
          pf_in,
          config=pf_cfg,
          two_m=two_m,
          opt_tol=opt_tol_arr,
          norm_kind=norm_kind,
          jit=True,
      )

      u_cand = tree.scale(two_m, pf_out.r)
      v_cand = tree.add(
          u_cand,
          tree.scale(two_m, tree.sub(st.center, pf_out.y)),
      )
      cand_opt = tree.norm(v_cand, kind=norm_kind)
      cand_dual = pen_oracle.effective_dual(pf_out.y)
      p_bar = tree.scale(1.0 - dampen, st.multiplier)
      al_feas = tree.norm(
          tree.scale(1.0 / st.penalty, tree.sub(cand_dual, p_bar)),
          kind=norm_kind,
      )
      raw_feas = tree.norm(bound_c.residual(pf_out.y), kind=norm_kind)
      cand_feas = jnp.maximum(al_feas, raw_feas)
      cand_finite = jnp.logical_and(
          tree.all_finite(pf_out.y),
          jnp.logical_and(jnp.isfinite(cand_opt), jnp.isfinite(cand_feas)),
      )
      cand_score = jnp.maximum(
          cand_opt / jnp.maximum(opt_tol_arr, tiny),
          cand_feas / jnp.maximum(feas_tol_arr, tiny),
      )
      prev_score = jnp.maximum(
          st.opt_residual / jnp.maximum(opt_tol_arr, tiny),
          st.feas_residual / jnp.maximum(feas_tol_arr, tiny),
      )
      is_better = jnp.logical_and(cand_finite, cand_score <= prev_score)
      best_x = jax.tree.map(
          lambda cy, bx: jnp.where(is_better, cy, bx),
          pf_out.y,
          st.best_x,
      )
      best_dual = jax.tree.map(
          lambda cd, bd: jnp.where(is_better, cd, bd),
          cand_dual,
          st.dual,
      )
      best_v = jax.tree.map(
          lambda cv, bv: jnp.where(is_better, cv, bv),
          v_cand,
          st.subgrad,
      )
      best_opt = jnp.where(is_better, cand_opt, st.opt_residual)
      best_feas = jnp.where(is_better, cand_feas, st.feas_residual)
      conv_now = jnp.logical_and(
          best_opt <= opt_tol_arr,
          best_feas <= feas_tol_arr,
      )
      still_running = jnp.logical_and(
          pf_out.status == running_code,
          jnp.logical_not(conv_now),
      )

      def _continue_inner() -> IALMTransformState:
        return dataclasses.replace(
            st,
            inner_iter=pf_out.iteration,
            total_steps=st.total_steps + 1,
            best_x=best_x,
            pf_x=pf_out.x,
            pf_y=pf_out.y,
            pf_r=pf_out.r,
            pf_a_sum=pf_out.a_sum,
            pf_psi_y=pf_out.psi_y,
            pf_lipschitz=pf_out.lipschitz,
            dual=best_dual,
            subgrad=best_v,
            opt_residual=best_opt,
            feas_residual=best_feas,
        )

      def _finish_subproblem() -> IALMTransformState:
        phi_y = pen_oracle.value(pf_out.y)
        eps_val = tree.eps(dtype)
        eps_sq = eps_val**2
        phi_slack = 100.0 * eps_val * (1.0 + jnp.abs(st.center_obj))
        desc_ok = (
            tree.norm_sq(v_cand)
            <= 2.0 * theta * m_hat * (st.center_obj - phi_y + phi_slack)
            + eps_sq
        )
        dist_sq = tree.norm_sq(tree.sub(pf_out.y, st.center))
        prox_err_ok = (
            tree.norm_sq(u_cand)
            <= 2.0 * ((eff_rho * m_hat) ** 2) * dist_sq + eps_sq
        )
        bad_code = jnp.asarray(
            int(pf_acg.PFACGStatus.BAD_TERMINATION),
            dtype=jnp.int32,
        )
        ls_fail_code = jnp.asarray(
            int(pf_acg.PFACGStatus.LINE_SEARCH_FAILED),
            dtype=jnp.int32,
        )
        stage_opt_done = cand_opt <= opt_tol_arr
        rejected = jnp.logical_and(
            jnp.logical_not(jnp.logical_or(conv_now, stage_opt_done)),
            jnp.logical_and(
                pf_out.status != ls_fail_code,
                jnp.logical_or(
                    pf_out.status == bad_code,
                    jnp.logical_not(jnp.logical_and(desc_ok, prox_err_ok)),
                ),
            ),
        )
        M_from_acg = jnp.maximum(
            jnp.asarray(1e-8, dtype=dtype),
            two_m * (pf_out.lipschitz - 1.0),
        )

        def _reject_outer() -> IALMTransformState:
          grown_m = jnp.minimum(m_hat * alpha, m_max_arr)
          M_keep = jnp.maximum(st.upper_curvature, M_from_acg)
          psi_y0_next = st.center_obj / (2.0 * grown_m)
          return dataclasses.replace(
              st,
              stage_outer_iter=st.stage_outer_iter + 1,
              outer_iter=st.outer_iter + 1,
              inner_iter=jnp.asarray(0, dtype=jnp.int32),
              total_steps=st.total_steps + 1,
              best_x=best_x,
              pf_x=st.center,
              pf_y=st.center,
              pf_r=tree.zeros_like(st.center),
              pf_a_sum=jnp.asarray(0.0, dtype=dtype),
              pf_psi_y0=psi_y0_next,
              pf_psi_y=psi_y0_next,
              pf_lipschitz=_sub_lip_init(grown_m, M_keep),
              lower_curvature=grown_m,
              upper_curvature=M_keep,
              never_increased_m=jnp.asarray(False, dtype=jnp.bool_),
              dual=best_dual,
              subgrad=best_v,
              opt_residual=best_opt,
              feas_residual=best_feas,
          )

        def _accept_outer() -> IALMTransformState:
          quad_curv = 0.5 * M_from_acg * dist_sq
          noise_floor = 100.0 * eps_val * (1.0 + jnp.abs(phi_y))
          M_est = jnp.where(
              quad_curv > noise_floor,
              M_from_acg,
              jnp.maximum(M_from_acg, st.upper_curvature),
          )
          if decay_curvature:
            m_relaxed = jnp.maximum(m_min_arr, m_hat / (1.0 + 0.5 * alpha))
            m_next = jnp.where(dist_sq > eps_sq, m_relaxed, m_hat)
          else:
            m_bitonic = jnp.maximum(m_min_arr, m_hat / alpha)
            m_next = jnp.where(st.never_increased_m, m_bitonic, m_hat)

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
              best_x,
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
          psi_y0_next = next_center_obj / (2.0 * m_next)
          return IALMTransformState(
              initialized=jnp.asarray(True, dtype=jnp.bool_),
              converged=conv_now,
              alm_iter=jnp.where(
                  advance_alm,
                  st.alm_iter + 1,
                  st.alm_iter,
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
              best_x=best_x,
              pf_x=next_center,
              pf_y=next_center,
              pf_r=tree.zeros_like(next_center),
              pf_a_sum=jnp.asarray(0.0, dtype=dtype),
              pf_psi_y0=psi_y0_next,
              pf_psi_y=psi_y0_next,
              pf_lipschitz=_sub_lip_init(m_next, M_next),
              multiplier=next_mult,
              penalty=next_penalty,
              prev_feas_res=prev_feas_next,
              lower_curvature=m_next,
              upper_curvature=M_next,
              never_increased_m=st.never_increased_m,
              dual=best_dual,
              subgrad=best_v,
              opt_residual=best_opt,
              feas_residual=best_feas,
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
