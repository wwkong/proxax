"""Indicator and constrained-norm proximal operators via `optax.projections`."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import optax

from proxax import _typing
from proxax import tree


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class NonNegative:
  """Indicator function of the non-negative orthant `x >= 0`."""

  @property
  def strong_convexity(self) -> float:
    """Strong convexity constant (`0.0`)."""
    return 0.0

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Returns `0.0` at feasible points."""
    return jnp.asarray(0.0, dtype=tree.infer_dtype(x))

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Projects `x` onto `x >= 0`."""
    del step_size
    return optax.projections.projection_non_negative(x)

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Projects `x` onto `x >= 0`."""
    return optax.projections.projection_non_negative(x)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Box:
  """Indicator function of a box `lower <= x <= upper`."""

  lower: _typing.PyTree = -jnp.inf
  upper: _typing.PyTree = jnp.inf

  @property
  def strong_convexity(self) -> float:
    """Strong convexity constant (`0.0`)."""
    return 0.0

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Returns `0.0` at feasible points."""
    return jnp.asarray(0.0, dtype=tree.infer_dtype(x))

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Clips `x` onto `[lower, upper]`."""
    del step_size
    return tree.clip_box(x, self.lower, self.upper)

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Clips `x` onto `[lower, upper]`."""
    return tree.clip_box(x, self.lower, self.upper)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class BoxL1:
  """Composite L1 regularizer restricted to a box `[lower, upper]`."""

  l1_weight: _typing.Scalar = 1.0
  lower: _typing.PyTree = -jnp.inf
  upper: _typing.PyTree = jnp.inf

  @property
  def strong_convexity(self) -> float:
    """Strong convexity constant (`0.0`)."""
    return 0.0

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates `l1_weight * ||x||_1` on `[lower, upper]`."""
    leaves = jax.tree.leaves(x)
    if not leaves:
      return jnp.asarray(0.0)
    total = tree.functools_sum([jnp.sum(jnp.abs(leaf)) for leaf in leaves])
    return self.l1_weight * total

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Applies soft-thresholding followed by box clipping."""
    thresh = step_size * self.l1_weight
    shrunk = jax.tree.map(
        lambda u: jnp.sign(u) * jnp.maximum(jnp.abs(u) - thresh, 0.0),
        x,
    )
    return tree.clip_box(shrunk, self.lower, self.upper)

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Clips `x` onto `[lower, upper]`."""
    return tree.clip_box(x, self.lower, self.upper)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class Simplex:
  """Indicator of the scaled probability simplex `{x >= 0 : sum(x) = scale}`."""

  scale: _typing.Scalar = 1.0

  @property
  def strong_convexity(self) -> float:
    """Strong convexity constant (`0.0`)."""
    return 0.0

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Returns `0.0` at feasible points."""
    return jnp.asarray(0.0, dtype=tree.infer_dtype(x))

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Projects `x` onto the simplex of radius `scale`."""
    del step_size
    return optax.projections.projection_simplex(x, self.scale)

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Projects `x` onto the simplex of radius `scale`."""
    return optax.projections.projection_simplex(x, self.scale)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class L2Ball:
  """Indicator of the Euclidean ball `{x : ||x||_2 <= radius}`."""

  radius: _typing.Scalar = 1.0

  @property
  def strong_convexity(self) -> float:
    """Strong convexity constant (`0.0`)."""
    return 0.0

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Returns `0.0` at feasible points."""
    return jnp.asarray(0.0, dtype=tree.infer_dtype(x))

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Projects `x` onto the L2 ball of radius `radius`."""
    del step_size
    return optax.projections.projection_l2_ball(x, self.radius)

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Projects `x` onto the L2 ball of radius `radius`."""
    return optax.projections.projection_l2_ball(x, self.radius)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class L1Ball:
  """Indicator of the L1 ball `{x : ||x||_1 <= radius}`."""

  radius: _typing.Scalar = 1.0

  @property
  def strong_convexity(self) -> float:
    """Strong convexity constant (`0.0`)."""
    return 0.0

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Returns `0.0` at feasible points."""
    return jnp.asarray(0.0, dtype=tree.infer_dtype(x))

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Projects `x` onto the L1 ball of radius `radius`."""
    del step_size
    return optax.projections.projection_l1_ball(x, self.radius)

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Projects `x` onto the L1 ball of radius `radius`."""
    return optax.projections.projection_l1_ball(x, self.radius)
