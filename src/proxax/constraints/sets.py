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


def project_exp_cone(v: jax.Array) -> jax.Array:
  """Projects a 3D vector `(v0, v1, v2)` onto the primal exponential cone."""
  v0, v1, v2 = v[0], v[1], v[2]
  dtype = v.dtype
  zero = jnp.zeros_like(v)

  in_primal = jnp.logical_or(
      jnp.logical_and(
          v1 > 0.0,
          v1 * jnp.exp(jnp.clip(v0 / jnp.maximum(v1, 1e-30), -40.0, 40.0))
          <= v2,
      ),
      jnp.logical_and(jnp.logical_and(v0 <= 0.0, v1 == 0.0), v2 >= 0.0),
  )
  in_polar = jnp.logical_or(
      jnp.logical_and(
          v0 > 0.0,
          v0 * jnp.exp(jnp.clip(v1 / jnp.maximum(v0, 1e-30), -40.0, 40.0))
          <= -jnp.e * v2,
      ),
      jnp.logical_and(jnp.logical_and(v0 == 0.0, v1 <= 0.0), v2 <= 0.0),
  )

  # Boundary ray: {x0 <= 0, x1 = 0, x2 >= 0}
  p_ray = jnp.stack([
      jnp.minimum(v0, 0.0),
      jnp.asarray(0.0, dtype=dtype),
      jnp.maximum(v2, 0.0),
  ])
  d_ray_sq = jnp.sum((p_ray - v) ** 2)

  # Smooth boundary KKT root in r = x0 / x1
  safe_v0 = jnp.where(jnp.abs(v0) > 1e-30, v0, 1.0)
  safe_v1 = jnp.where(jnp.abs(v1) > 1e-30, v1, 1.0)
  lb_x1 = jnp.where(v0 > 0.0, 1.0 - v1 / safe_v0, -30.0)
  lb_lam = jnp.where(v1 < 0.0, v0 / safe_v1, -30.0)
  r_lo = jnp.maximum(-30.0, jnp.maximum(lb_x1, lb_lam))

  ub_x1 = jnp.where(v0 < 0.0, 1.0 - v1 / safe_v0, 30.0)
  ub_lam = jnp.where(v1 > 0.0, v0 / safe_v1, 30.0)
  r_hi = jnp.minimum(30.0, jnp.minimum(ub_x1, ub_lam))
  r_hi = jnp.maximum(r_lo, r_hi)

  def _h(r: jax.Array) -> jax.Array:
    er = jnp.exp(jnp.clip(r, -30.0, 30.0))
    emr = jnp.exp(jnp.clip(-r, -30.0, 30.0))
    return (
        (v1 + (r - 1.0) * v0) * er
        - v2 * (1.0 - r + r * r)
        - (v0 - r * v1) * emr
    )

  def _bisect_step(_: int, bounds: tuple[jax.Array, jax.Array]) -> tuple[
      jax.Array, jax.Array
  ]:
    lo, hi = bounds
    mid = 0.5 * (lo + hi)
    go_right = _h(mid) < 0.0
    return (jnp.where(go_right, mid, lo), jnp.where(go_right, hi, mid))

  lo_f, hi_f = jax.lax.fori_loop(0, 60, _bisect_step, (r_lo, r_hi))
  r_star = 0.5 * (lo_f + hi_f)
  denom = 1.0 - r_star + r_star * r_star
  x1_s = jnp.maximum((v1 + (r_star - 1.0) * v0) / denom, 0.0)
  x0_s = r_star * x1_s
  x2_s = x1_s * jnp.exp(jnp.clip(r_star, -30.0, 30.0))
  p_smooth = jnp.stack([x0_s, x1_s, x2_s])
  d_smooth_sq = jnp.sum((p_smooth - v) ** 2)

  valid_smooth = jnp.logical_and(r_lo < r_hi, x1_s > 0.0)
  p_bdy = jnp.where(
      jnp.logical_and(valid_smooth, d_smooth_sq < d_ray_sq),
      p_smooth,
      p_ray,
  )
  return jnp.where(in_primal, v, jnp.where(in_polar, zero, p_bdy))


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class ExponentialCone:
  """Primal 3D exponential cone indicator set."""

  def project(self, y: _typing.PyTree) -> _typing.PyTree:
    """Projects each 3-vector leaf onto the primal exponential cone."""
    return jax.tree.map(project_exp_cone, y)
