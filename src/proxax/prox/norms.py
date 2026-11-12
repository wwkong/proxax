"""Norm-based proximal regularizers (`L1`, `L2`, `L2Squared`, `ElasticNet`)."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from proxax import _typing
from proxax import tree


def _soft_threshold(u: jax.Array, thresh: _typing.Scalar) -> jax.Array:
  """Elementwise soft-thresholding `sign(u) * max(|u| - thresh, 0)`."""
  return jnp.sign(u) * jnp.maximum(jnp.abs(u) - thresh, 0.0)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class L1:
  """Weighted L1 norm `h(x) = weight * ||x||_1`."""

  weight: _typing.Scalar = 1.0

  @property
  def strong_convexity(self) -> float:
    """Strong convexity constant (`0.0`)."""
    return 0.0

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates `weight * sum_i |x_i|` across all leaves."""
    leaves = jax.tree.leaves(x)
    if not leaves:
      return jnp.asarray(0.0)
    total = tree.functools_sum([jnp.sum(jnp.abs(leaf)) for leaf in leaves])
    return self.weight * total

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Applies elementwise soft-thresholding with `step_size * weight`."""
    thresh = step_size * self.weight
    return jax.tree.map(lambda leaf: _soft_threshold(leaf, thresh), x)

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Identity projection (`dom(L1)` is the full space)."""
    return x


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class L2Squared:
  """Half squared Euclidean norm `h(x) = (weight / 2) * ||x||_2^2`."""

  weight: _typing.Scalar = 1.0

  @property
  def strong_convexity(self) -> _typing.Scalar:
    """Strong convexity constant equal to `weight`."""
    return self.weight

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates `0.5 * weight * ||x||_2^2`."""
    return 0.5 * self.weight * tree.norm_sq(x)

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Scales `x` by `1 / (1 + step_size * weight)`."""
    inv = 1.0 / (1.0 + step_size * self.weight)
    return tree.scale(inv, x)

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Identity projection."""
    return x


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class L2:
  """Unsquared Euclidean (group) norm `h(x) = weight * ||x||_2`."""

  weight: _typing.Scalar = 1.0

  @property
  def strong_convexity(self) -> float:
    """Strong convexity constant (`0.0`)."""
    return 0.0

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates `weight * ||x||_2`."""
    return self.weight * tree.l2_norm(x)

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Applies Euclidean block soft-thresholding with `step_size * weight`."""
    thresh = step_size * self.weight
    norm_x = tree.l2_norm(x)
    safe_norm = jnp.where(norm_x > 0.0, norm_x, 1.0)
    factor = jnp.where(norm_x > thresh, 1.0 - thresh / safe_norm, 0.0)
    return tree.scale(factor, x)

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Identity projection."""
    return x


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ElasticNet:
  """Elastic-net `l1_weight * ||x||_1 + (l2_weight / 2) * ||x||_2^2`."""

  l1_weight: _typing.Scalar = 1.0
  l2_weight: _typing.Scalar = 1.0

  @property
  def strong_convexity(self) -> _typing.Scalar:
    """Strong convexity constant equal to `l2_weight`."""
    return self.l2_weight

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates elastic-net objective value."""
    leaves = jax.tree.leaves(x)
    if not leaves:
      return jnp.asarray(0.0)
    l1_val = tree.functools_sum([jnp.sum(jnp.abs(leaf)) for leaf in leaves])
    return self.l1_weight * l1_val + 0.5 * self.l2_weight * tree.norm_sq(x)

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Applies soft-thresholding followed by L2 shrinkage."""
    thresh = step_size * self.l1_weight
    inv = 1.0 / (1.0 + step_size * self.l2_weight)
    return jax.tree.map(lambda leaf: inv * _soft_threshold(leaf, thresh), x)

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Identity projection."""
    return x
