"""Optax gradient transformations for `proxax` operators and solvers."""

from __future__ import annotations

import dataclasses
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
