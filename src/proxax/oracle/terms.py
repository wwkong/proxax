"""Smooth objective terms and curvature metadata."""

from __future__ import annotations

import dataclasses
import typing

import jax
import jax.numpy as jnp

from proxax import _typing
from proxax import tree


@typing.runtime_checkable
class SmoothObjective(typing.Protocol):
  """Protocol for a differentiable term `f` with optional curvatures."""

  @property
  def curvature_upper(self) -> jax.Array | float | None:
    """Optional global smoothness upper bound `L^+ >= 0`."""
    ...

  @property
  def curvature_lower(self) -> jax.Array | float | None:
    """Optional weak-convexity / negative-curvature bound `m = L^- >= 0`."""
    ...

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates `f(x)`."""
    ...

  def value_and_grad(
      self,
      x: _typing.PyTree,
  ) -> tuple[jax.Array, _typing.PyTree]:
    """Evaluates `(f(x), grad_f(x))`."""
    ...


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class ZeroSmooth:
  """Identically zero smooth objective `f(x) = 0`."""

  @property
  def curvature_upper(self) -> float:
    """Upper smoothness constant (`0.0`)."""
    return 0.0

  @property
  def curvature_lower(self) -> float:
    """Lower nonconvexity constant (`0.0`)."""
    return 0.0

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Returns scalar `0.0` matching the dtype of `x`."""
    return jnp.asarray(0.0, dtype=tree.infer_dtype(x))

  def value_and_grad(
      self,
      x: _typing.PyTree,
  ) -> tuple[jax.Array, _typing.PyTree]:
    """Returns `(0.0, zeros_like(x))`."""
    return jnp.asarray(0.0, dtype=tree.infer_dtype(x)), tree.zeros_like(x)


@dataclasses.dataclass(frozen=True, slots=True)
class SmoothTerm:
  """Wraps a smooth callable `fn` with optional custom grad and curvatures.

  Attributes:
    fn: Scalar smooth function `x -> f(x)`.
    vag_fn: Optional custom `(value, grad)` callable; defaults to
      `jax.value_and_grad(fn)`.
    upper_curvature: Optional global smoothness constant `L^+ >= 0`.
    lower_curvature: Optional global weak-convexity constant `m = L^- >= 0`
      such that `f(x) + (m / 2) ||x||_2^2` is convex.
  """

  fn: _typing.SmoothFn
  _: dataclasses.KW_ONLY
  vag_fn: _typing.ValueAndGradFn | None = None
  upper_curvature: jax.Array | float | None = None
  lower_curvature: jax.Array | float | None = None

  @property
  def curvature_upper(self) -> jax.Array | float | None:
    """Optional smoothness upper bound `L^+`."""
    return self.upper_curvature

  @property
  def curvature_lower(self) -> jax.Array | float | None:
    """Optional weak-convexity constant `m = L^-`."""
    return self.lower_curvature

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates `fn(x)` as a JAX scalar array."""
    return jnp.asarray(self.fn(x), dtype=tree.infer_dtype(x))

  def value_and_grad(
      self,
      x: _typing.PyTree,
  ) -> tuple[jax.Array, _typing.PyTree]:
    """Evaluates `(f(x), grad_f(x))`."""
    if self.vag_fn is not None:
      val, grad = self.vag_fn(x)
      return jnp.asarray(val, dtype=tree.infer_dtype(x)), grad
    val, grad = jax.value_and_grad(self.fn)(x)
    return jnp.asarray(val, dtype=tree.infer_dtype(x)), grad


jax.tree_util.register_dataclass(
    SmoothTerm,
    data_fields=["upper_curvature", "lower_curvature"],
    meta_fields=["fn", "vag_fn"],
)


SmoothLike: typing.TypeAlias = SmoothObjective | _typing.SmoothFn | None


def as_smooth_term(
    smooth: SmoothLike,
    *,
    upper_curvature: jax.Array | float | None = None,
    lower_curvature: jax.Array | float | None = None,
) -> SmoothObjective:
  """Coerces `None`, a callable, or `SmoothObjective` to `SmoothObjective`."""
  if smooth is None:
    return ZeroSmooth()
  if isinstance(smooth, SmoothObjective):
    return smooth
  if callable(smooth):
    return SmoothTerm(
        smooth,
        upper_curvature=upper_curvature,
        lower_curvature=lower_curvature,
    )
  raise TypeError(f"Unsupported smooth objective type: {type(smooth)!r}.")
