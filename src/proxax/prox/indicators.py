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


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class LInfBall:
  """Indicator of the L-infinity ball `{x : ||x||_inf <= radius}`."""

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
    """Clips `x` onto `[-radius, radius]`."""
    del step_size
    return tree.clip_box(x, -self.radius, self.radius)

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Clips `x` onto `[-radius, radius]`."""
    return tree.clip_box(x, -self.radius, self.radius)


def _project_capped_simplex(
    w: jax.Array,
    k: _typing.Scalar,
) -> jax.Array:
  """Projects vector(s) `w` onto `{z in [0, 1]^n : sum(z) = k}`."""
  lo = jnp.min(w, axis=-1, keepdims=True) - 1.0
  hi = jnp.max(w, axis=-1, keepdims=True)

  def _bisect_step(
      _: int,
      bounds: tuple[jax.Array, jax.Array],
  ) -> tuple[jax.Array, jax.Array]:
    cur_lo, cur_hi = bounds
    mid = 0.5 * (cur_lo + cur_hi)
    total = jnp.sum(jnp.clip(w - mid, 0.0, 1.0), axis=-1, keepdims=True)
    next_lo = jnp.where(total > k, mid, cur_lo)
    next_hi = jnp.where(total > k, cur_hi, mid)
    return next_lo, next_hi

  final_lo, final_hi = jax.lax.fori_loop(0, 60, _bisect_step, (lo, hi))
  nu = 0.5 * (final_lo + final_hi)
  return jnp.clip(w - nu, 0.0, 1.0)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class Fantope:
  """Indicator of the k-Fantope `{X = X^T : 0 <= X <= I, tr(X) = k}`."""

  k: _typing.Scalar = 1.0

  @property
  def strong_convexity(self) -> float:
    """Strong convexity constant (`0.0`)."""
    return 0.0

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Returns `0.0` at feasible points."""
    return jnp.asarray(0.0, dtype=tree.infer_dtype(x))

  def _project(self, x: _typing.PyTree) -> _typing.PyTree:
    def _proj_leaf(mat: jax.Array) -> jax.Array:
      sym = 0.5 * (mat + jnp.swapaxes(mat, -1, -2))
      eigvals, eigvecs = jnp.linalg.eigh(sym)
      eigvals_proj = _project_capped_simplex(eigvals, self.k)
      return jnp.asarray(
          (eigvecs * eigvals_proj[..., None, :])
          @ jnp.swapaxes(eigvecs, -1, -2)
      )

    return jax.tree.map(_proj_leaf, x)

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Projects `x` onto the k-Fantope."""
    del step_size
    return self._project(x)

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Projects `x` onto the k-Fantope."""
    return self._project(x)
