"""Immutable composite optimization problem definition and KKT diagnostics."""

from __future__ import annotations

import dataclasses
import typing

import jax
import jax.numpy as jnp

from proxax import _typing
from proxax import tree
from proxax.constraints import linear_op
from proxax.oracle import composite
from proxax.oracle import terms
from proxax.prox import base as prox_base


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class KKTResiduals:
  """Evaluated optimality and feasibility residuals at a candidate solution."""

  primal_residual_norm: jax.Array
  stationarity_norm: jax.Array
  objective_value: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class CompositeProblem:
  """Immutable problem specification `min_x f(x) + h(x) s.t. A(x) in C`.

  Attributes:
    smooth: Smooth differentiable objective `f` (`Callable`, `SmoothTerm`, or
      `None` for `0`).
    proximal: Convex `ProxOperator` `h` (defaults to `prox.Zero()`).
    constraints: Optional `Constraint` or PyTree (e.g. `dict` / `tuple`) of
      `Constraint`s defining `A(x) in C`.
  """

  smooth: terms.SmoothLike = dataclasses.field(default_factory=terms.ZeroSmooth)
  proximal: prox_base.ProxOperator = dataclasses.field(
      default_factory=prox_base.Zero
  )
  constraints: linear_op.Constraint | _typing.PyTree | None = None

  def __post_init__(self) -> None:
    """Normalizes `smooth` and `constraints` fields on construction."""
    if self.smooth is None or (
        callable(self.smooth)
        and not isinstance(self.smooth, terms.SmoothObjective)
    ):
      object.__setattr__(self, "smooth", terms.as_smooth_term(self.smooth))
    if isinstance(self.constraints, dict | list | tuple):
      normalized = linear_op.normalize_constraints(self.constraints)
      object.__setattr__(self, "constraints", normalized)

  @property
  def is_constrained(self) -> bool:
    """Returns True if the problem includes set/cone constraints `A(x) in C`."""
    return self.constraints is not None

  @property
  def oracle(self) -> composite.CompositeOracle:
    """Returns the base unconstrained `CompositeOracle` for `f + h`."""
    return composite.CompositeOracle(
        smooth=self.smooth,
        proximal=self.proximal,
    )

  def bind_constraints(
      self,
      x_example: _typing.PyTree,
  ) -> linear_op.BoundConstraint | None:
    """Binds `constraints` to `x_example`'s PyTree structure if present."""
    if self.constraints is None:
      return None
    c_obj = typing.cast(linear_op.Constraint, self.constraints)
    return c_obj.bind(x_example)

  def evaluate_residuals(
      self,
      x: _typing.PyTree,
      subgrad: _typing.PyTree,
      *,
      norm_kind: str = "l2",
  ) -> KKTResiduals:
    """Computes objective value, stationarity norm, and feasibility residual."""
    dtype = tree.infer_dtype(x)
    obj = self.oracle.value(x)
    stat_norm = tree.norm(subgrad, kind=norm_kind)
    bound_c = self.bind_constraints(x)
    if bound_c is None:
      feas_norm = jnp.asarray(0.0, dtype=dtype)
    else:
      feas_norm = tree.norm(bound_c.residual(x), kind=norm_kind)
    return KKTResiduals(
        primal_residual_norm=feas_norm,
        stationarity_norm=stat_norm,
        objective_value=obj,
    )
