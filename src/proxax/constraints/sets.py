"""Closed convex sets equipped with Euclidean projections."""

from __future__ import annotations

import dataclasses
import typing

import jax
import jax.numpy as jnp

from proxax import _typing
from proxax import tree


@typing.runtime_checkable
class ConvexSet(typing.Protocol):
  """Protocol for a closed convex set `C` with Euclidean projection `Pi_C`."""

  def project(self, y: _typing.PyTree) -> _typing.PyTree:
    """Computes `argmin_{z in C} ||z - y||_2^2`."""
    ...


def _broadcast_rhs(y_leaf: jax.Array, rhs_leaf: typing.Any) -> jax.Array:
  rhs_arr = jnp.asarray(rhs_leaf, dtype=y_leaf.dtype)
  return jnp.broadcast_to(rhs_arr, y_leaf.shape)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class ZeroSet:
  """Singleton origin `{0}` for homogeneous equality constraints `A(x) = 0`."""

  def project(self, y: _typing.PyTree) -> _typing.PyTree:
    """Projects onto `0`."""
    return tree.zeros_like(y)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class Singleton:
  """Singleton `{rhs}` representing equality constraints `A(x) = rhs`."""

  rhs: _typing.PyTree

  def project(self, y: _typing.PyTree) -> _typing.PyTree:
    """Returns `rhs` broadcast to the leaves of `y`."""
    if (
        not isinstance(self.rhs, dict | list | tuple)
        and jnp.ndim(self.rhs) == 0
    ):
      return jax.tree.map(lambda leaf: _broadcast_rhs(leaf, self.rhs), y)
    return jax.tree.map(_broadcast_rhs, y, self.rhs)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class UpperBound:
  """Halfspace/orthant `{z : z <= rhs}` for `A(x) <= rhs`."""

  rhs: _typing.PyTree = 0.0

  def project(self, y: _typing.PyTree) -> _typing.PyTree:
    """Computes `min(y, rhs)` elementwise."""
    if (
        not isinstance(self.rhs, dict | list | tuple)
        and jnp.ndim(self.rhs) == 0
    ):
      return jax.tree.map(
          lambda leaf: jnp.minimum(leaf, _broadcast_rhs(leaf, self.rhs)),
          y,
      )
    return jax.tree.map(
        lambda leaf, r: jnp.minimum(leaf, _broadcast_rhs(leaf, r)),
        y,
        self.rhs,
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class LowerBound:
  """Halfspace/orthant `{z : z >= rhs}` for `A(x) >= rhs`."""

  rhs: _typing.PyTree = 0.0

  def project(self, y: _typing.PyTree) -> _typing.PyTree:
    """Computes `max(y, rhs)` elementwise."""
    if (
        not isinstance(self.rhs, dict | list | tuple)
        and jnp.ndim(self.rhs) == 0
    ):
      return jax.tree.map(
          lambda leaf: jnp.maximum(leaf, _broadcast_rhs(leaf, self.rhs)),
          y,
      )
    return jax.tree.map(
        lambda leaf, r: jnp.maximum(leaf, _broadcast_rhs(leaf, r)),
        y,
        self.rhs,
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class BoxSet:
  """Box `{z : lower <= z <= upper}` for two-sided bounds on `A(x)`."""

  lower: _typing.PyTree = -jnp.inf
  upper: _typing.PyTree = jnp.inf

  def project(self, y: _typing.PyTree) -> _typing.PyTree:
    """Clips `y` onto `[lower, upper]`."""
    return tree.clip_box(y, self.lower, self.upper)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class SecondOrderCone:
  """Lorentz / second-order cone `{(t, u) : ||u||_2 <= t}` on 1D arrays."""

  def project(self, y: jax.Array) -> jax.Array:
    """Projects a 1D vector `y = [t, u_1, ..., u_d]` onto the SOC."""
    t = y[0]
    u = y[1:]
    u_norm = jnp.linalg.norm(u)
    safe_norm = jnp.where(u_norm > 0.0, u_norm, 1.0)
    scale = 0.5 * (t + u_norm)
    mid_proj = jnp.concatenate([
        jnp.expand_dims(scale, axis=0),
        (scale / safe_norm) * u,
    ])
    return jnp.where(
        u_norm <= t,
        y,
        jnp.where(u_norm <= -t, jnp.zeros_like(y), mid_proj),
    )


def _is_convex_set(node: typing.Any) -> bool:
  return isinstance(node, ConvexSet)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class ProductSet:
  """Cartesian product of `ConvexSet` objects arranged as a PyTree."""

  sets: _typing.PyTree

  def project(self, y: _typing.PyTree) -> _typing.PyTree:
    """Projects each subtree of `y` onto its corresponding `ConvexSet`."""
    return jax.tree.map(
        lambda s, sub_y: s.project(sub_y),
        self.sets,
        y,
        is_leaf=_is_convex_set,
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class PSDSet:
  """Symmetric positive semidefinite cone `{Z : Z = Z^T, Z >= 0}`."""

  def project(self, y: _typing.PyTree) -> _typing.PyTree:
    """Projects each symmetric matrix leaf onto the PSD cone."""

    def _proj_leaf(mat: jax.Array) -> jax.Array:
      sym = 0.5 * (mat + jnp.swapaxes(mat, -1, -2))
      eigvals, eigvecs = jnp.linalg.eigh(sym)
      clipped = jnp.maximum(eigvals, 0.0)
      return jnp.asarray(
          (eigvecs * clipped[..., None, :]) @ jnp.swapaxes(eigvecs, -1, -2)
      )

    return jax.tree.map(_proj_leaf, y)
