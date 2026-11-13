"""Composable Optax tracker states and `proximal` transform."""

from __future__ import annotations

import dataclasses
import typing
from collections import abc

import jax
import jax.numpy as jnp
import optax

from proxax import _typing
from proxax import tree
from proxax.prox import base as prox_base

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


# ---------------------------------------------------------------------------
# Composable trackers, subproblem carries, and zero-slot @property mixins
# ---------------------------------------------------------------------------


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class OptaxTracker:
  """Composable outer lifecycle and certificate tracker for Optax transforms.

  Attributes:
    initialized: Scalar bool indicating whether step-0 bootstrap has run.
    converged: Scalar bool indicating whether `opt_residual <= opt_tol`.
    outer_iter: Completed outer proximal iterations.
    inner_iter: Current inner subproblem step counter.
    total_steps: Total `tx.update` calls executed.
    center: Active proximal center `x_k^-`.
    center_obj: Composite objective value at `center`.
    best_x: Best certified primal iterate `x_hat` seen so far.
    subgrad: Certified composite subgradient `v_hat` at `best_x`.
    opt_residual: Stationarity residual norm `||v_hat||` at `best_x`.
  """

  initialized: jax.Array
  converged: jax.Array
  outer_iter: jax.Array
  inner_iter: jax.Array
  total_steps: jax.Array
  center: _typing.PyTree
  center_obj: jax.Array
  best_x: _typing.PyTree
  subgrad: _typing.PyTree
  opt_residual: jax.Array

  @classmethod
  def uninitialized(cls, p0: _typing.PyTree) -> OptaxTracker:
    """Creates an uninitialized `OptaxTracker` matching `p0`."""
    dtype = tree.infer_dtype(p0)
    z0 = tree.zeros_like(p0)
    zero_i32 = jnp.asarray(0, dtype=jnp.int32)
    return cls(
        initialized=jnp.asarray(False, dtype=jnp.bool_),
        converged=jnp.asarray(False, dtype=jnp.bool_),
        outer_iter=zero_i32,
        inner_iter=zero_i32,
        total_steps=zero_i32,
        center=p0,
        center_obj=jnp.asarray(0.0, dtype=dtype),
        best_x=p0,
        subgrad=z0,
        opt_residual=jnp.asarray(jnp.inf, dtype=dtype),
    )

  def bootstrap(
      self,
      *,
      center: _typing.PyTree,
      center_obj: jax.Array,
      subgrad: _typing.PyTree,
      opt_residual: jax.Array,
      converged: jax.Array,
  ) -> OptaxTracker:
    """Populates step-0 refined certificate quantities."""
    return dataclasses.replace(
        self,
        initialized=jnp.asarray(True, dtype=jnp.bool_),
        converged=converged,
        inner_iter=jnp.asarray(0, dtype=jnp.int32),
        center=center,
        center_obj=center_obj,
        best_x=center,
        subgrad=subgrad,
        opt_residual=opt_residual,
    )

  def increment_step(self) -> OptaxTracker:
    """Increments `total_steps` by 1."""
    return dataclasses.replace(self, total_steps=self.total_steps + 1)

  def update_incumbent(
      self,
      x: _typing.PyTree,
      subgrad: _typing.PyTree,
      opt_res: jax.Array,
      *,
      check_finite: bool = True,
  ) -> OptaxTracker:
    """Updates `(best_x, subgrad, opt_residual)` when `opt_res` improves."""
    if check_finite:
      cand_finite = jnp.logical_and(tree.all_finite(x), jnp.isfinite(opt_res))
      is_better = jnp.logical_and(cand_finite, opt_res <= self.opt_residual)
    else:
      is_better = opt_res <= self.opt_residual
    best_x = jax.tree.map(
        lambda cx, bx: jnp.where(is_better, cx, bx),
        x,
        self.best_x,
    )
    best_v = jax.tree.map(
        lambda cv, bv: jnp.where(is_better, cv, bv),
        subgrad,
        self.subgrad,
    )
    best_res = jnp.where(is_better, opt_res, self.opt_residual)
    return dataclasses.replace(
        self,
        best_x=best_x,
        subgrad=best_v,
        opt_residual=best_res,
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ConstrainedOptaxTracker:
  """Composable outer KKT and homotopy tracker for constrained Optax transforms.

  Attributes:
    initialized: Scalar bool indicating whether step-0 bootstrap has run.
    converged: Scalar bool indicating KKT convergence.
    stage_iter: Completed outer penalty or Augmented Lagrangian stages.
    stage_outer_iter: Proximal outer steps completed in the current stage.
    outer_iter: Cumulative proximal outer steps across all stages.
    inner_iter: Current inner subproblem step counter.
    total_steps: Total `tx.update` calls executed.
    center: Active proximal center `x_k^-`.
    center_obj: Penalized/AL objective value at `center`.
    best_x: Best certified primal iterate `x_hat` seen so far.
    penalty: Active quadratic penalty parameter `c_k`.
    dual: Certified Lagrange multiplier `p_hat` at `best_x`.
    subgrad: Certified Lagrangian subgradient `v_hat` at `best_x`.
    opt_residual: Lagrangian stationarity norm `||v_hat||` at `best_x`.
    feas_residual: Primal feasibility / complementarity residual at `best_x`.
  """

  initialized: jax.Array
  converged: jax.Array
  stage_iter: jax.Array
  stage_outer_iter: jax.Array
  outer_iter: jax.Array
  inner_iter: jax.Array
  total_steps: jax.Array
  center: _typing.PyTree
  center_obj: jax.Array
  best_x: _typing.PyTree
  penalty: jax.Array
  dual: _typing.PyTree
  subgrad: _typing.PyTree
  opt_residual: jax.Array
  feas_residual: jax.Array

  @classmethod
  def uninitialized(
      cls,
      p0: _typing.PyTree,
      *,
      dual0: _typing.PyTree,
      penalty_init: float,
  ) -> ConstrainedOptaxTracker:
    """Creates an uninitialized `ConstrainedOptaxTracker`."""
    dtype = tree.infer_dtype(p0)
    z0 = tree.zeros_like(p0)
    zero_i32 = jnp.asarray(0, dtype=jnp.int32)
    inf_val = jnp.asarray(jnp.inf, dtype=dtype)
    return cls(
        initialized=jnp.asarray(False, dtype=jnp.bool_),
        converged=jnp.asarray(False, dtype=jnp.bool_),
        stage_iter=zero_i32,
        stage_outer_iter=zero_i32,
        outer_iter=zero_i32,
        inner_iter=zero_i32,
        total_steps=zero_i32,
        center=p0,
        center_obj=jnp.asarray(0.0, dtype=dtype),
        best_x=p0,
        penalty=jnp.asarray(penalty_init, dtype=dtype),
        dual=dual0,
        subgrad=z0,
        opt_residual=inf_val,
        feas_residual=inf_val,
    )

  def bootstrap(
      self,
      *,
      center: _typing.PyTree,
      center_obj: jax.Array,
      dual: _typing.PyTree,
      subgrad: _typing.PyTree,
      opt_residual: jax.Array,
      feas_residual: jax.Array,
      converged: jax.Array,
  ) -> ConstrainedOptaxTracker:
    """Populates step-0 refined KKT certificate quantities."""
    zero_i32 = jnp.asarray(0, dtype=jnp.int32)
    return dataclasses.replace(
        self,
        initialized=jnp.asarray(True, dtype=jnp.bool_),
        converged=converged,
        stage_outer_iter=zero_i32,
        inner_iter=zero_i32,
        center=center,
        center_obj=center_obj,
        best_x=center,
        dual=dual,
        subgrad=subgrad,
        opt_residual=opt_residual,
        feas_residual=feas_residual,
    )

  def increment_step(self) -> ConstrainedOptaxTracker:
    """Increments `total_steps` by 1."""
    return dataclasses.replace(self, total_steps=self.total_steps + 1)

  def update_incumbent(
      self,
      x: _typing.PyTree,
      dual: _typing.PyTree,
      subgrad: _typing.PyTree,
      opt_res: jax.Array,
      feas_res: jax.Array,
      *,
      opt_tol_arr: jax.Array,
      feas_tol_arr: jax.Array,
  ) -> ConstrainedOptaxTracker:
    """Updates KKT incumbent `(best_x, dual, subgrad)` by normalized score."""
    tiny = jnp.asarray(1e-30, dtype=opt_res.dtype)
    cand_finite = jnp.logical_and(
        tree.all_finite(x),
        jnp.logical_and(jnp.isfinite(opt_res), jnp.isfinite(feas_res)),
    )
    cand_score = jnp.maximum(
        opt_res / jnp.maximum(opt_tol_arr, tiny),
        feas_res / jnp.maximum(feas_tol_arr, tiny),
    )
    prev_score = jnp.maximum(
        self.opt_residual / jnp.maximum(opt_tol_arr, tiny),
        self.feas_residual / jnp.maximum(feas_tol_arr, tiny),
    )
    is_better = jnp.logical_and(cand_finite, cand_score <= prev_score)
    best_x = jax.tree.map(
        lambda cx, bx: jnp.where(is_better, cx, bx),
        x,
        self.best_x,
    )
    best_dual = jax.tree.map(
        lambda cd, bd: jnp.where(is_better, cd, bd),
        dual,
        self.dual,
    )
    best_v = jax.tree.map(
        lambda cv, bv: jnp.where(is_better, cv, bv),
        subgrad,
        self.subgrad,
    )
    best_opt = jnp.where(is_better, opt_res, self.opt_residual)
    best_feas = jnp.where(is_better, feas_res, self.feas_residual)
    return dataclasses.replace(
        self,
        best_x=best_x,
        dual=best_dual,
        subgrad=best_v,
        opt_residual=best_opt,
        feas_residual=best_feas,
    )

