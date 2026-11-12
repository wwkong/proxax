"""PyTree Hilbert-space operations, norms, and dtype utilities."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp
import numpy as np

from proxax import _typing


def eps(dtype: jnp.dtype[typing.Any]) -> float:
  """Returns floating-point machine epsilon for `dtype`."""
  return float(np.finfo(dtype).eps)


def infer_dtype(tree: _typing.PyTree) -> jnp.dtype[typing.Any]:
  """Infers a floating-point JAX dtype from the leaves of `tree`.

  Args:
    tree: Input PyTree of JAX arrays.

  Returns:
    Floating-point dtype of the first floating leaf, or default float dtype.
  """
  for leaf in jax.tree.leaves(tree):
    arr = jnp.asarray(leaf)
    if jnp.issubdtype(arr.dtype, jnp.floating):
      return arr.dtype
  return jnp.asarray(0.0).dtype


def as_float_tree(tree: _typing.PyTree) -> _typing.PyTree:
  """Converts leaves of `tree` to floating-point JAX arrays."""
  default_dtype = jnp.asarray(0.0).dtype

  def _to_float(leaf: typing.Any) -> jax.Array:
    arr = jnp.asarray(leaf)
    if jnp.issubdtype(arr.dtype, jnp.floating):
      return arr
    return arr.astype(default_dtype)

  return jax.tree.map(_to_float, tree)


def zeros_like(tree: _typing.PyTree) -> _typing.PyTree:
  """Returns a PyTree of zeros matching the structure, shapes, and dtypes."""
  return jax.tree.map(jnp.zeros_like, tree)


def add(x: _typing.PyTree, y: _typing.PyTree) -> _typing.PyTree:
  """Computes elementwise `x + y` across two matching PyTrees."""
  return jax.tree.map(jnp.add, x, y)


def sub(x: _typing.PyTree, y: _typing.PyTree) -> _typing.PyTree:
  """Computes elementwise `x - y` across two matching PyTrees."""
  return jax.tree.map(jnp.subtract, x, y)


def scale(scalar: _typing.Scalar, x: _typing.PyTree) -> _typing.PyTree:
  """Multiplies every leaf of `x` by `scalar`."""
  return jax.tree.map(lambda leaf: scalar * leaf, x)


def add_scaled(
    x: _typing.PyTree,
    scalar: _typing.Scalar,
    y: _typing.PyTree,
) -> _typing.PyTree:
  """Computes `x + scalar * y` across matching PyTrees."""
  return jax.tree.map(lambda a, b: a + scalar * b, x, y)


def convex_comb(
    w_x: _typing.Scalar,
    x: _typing.PyTree,
    w_y: _typing.Scalar,
    y: _typing.PyTree,
) -> _typing.PyTree:
  """Computes `(w_x * x + w_y * y) / (w_x + w_y)` across matching PyTrees."""
  denom = w_x + w_y
  return jax.tree.map(lambda a, b: (w_x * a + w_y * b) / denom, x, y)


def vdot(x: _typing.PyTree, y: _typing.PyTree) -> jax.Array:
  """Computes the real Euclidean inner product `<x, y>` over PyTrees."""
  x_leaves = jax.tree.leaves(x)
  y_leaves = jax.tree.leaves(y)
  if not x_leaves:
    return jnp.asarray(0.0)
  dots = [
      jnp.real(jnp.vdot(a, b))
      for a, b in zip(x_leaves, y_leaves, strict=True)
  ]
  return functools_sum(dots)


def functools_sum(arrays: list[jax.Array]) -> jax.Array:
  """Sums a non-empty list of scalar JAX arrays preserving dtype."""
  acc = arrays[0]
  for arr in arrays[1:]:
    acc = acc + arr
  return acc


def norm_sq(x: _typing.PyTree) -> jax.Array:
  """Computes the squared Euclidean norm `||x||_2^2`."""
  return vdot(x, x)


def l2_norm(x: _typing.PyTree) -> jax.Array:
  """Computes the Euclidean norm `||x||_2`."""
  return jnp.sqrt(norm_sq(x))


def linf_norm(x: _typing.PyTree) -> jax.Array:
  """Computes the infinity norm `||x||_inf` across all leaves."""
  leaves = jax.tree.leaves(x)
  if not leaves:
    return jnp.asarray(0.0)
  dtype = infer_dtype(x)
  maxes = [
      (
          jnp.max(jnp.abs(leaf))
          if jnp.size(leaf) > 0
          else jnp.asarray(0.0, dtype=dtype)
      )
      for leaf in leaves
  ]
  acc = maxes[0]
  for val in maxes[1:]:
    acc = jnp.maximum(acc, val)
  return acc


def norm(x: _typing.PyTree, kind: str = "l2") -> jax.Array:
  """Computes either `'l2'` or `'linf'` norm of a PyTree.

  Args:
    x: Input PyTree.
    kind: Either `'l2'` or `'linf'`.

  Returns:
    Scalar JAX array norm.

  Raises:
    ValueError: If `kind` is neither `'l2'` nor `'linf'`.
  """
  if kind == "l2":
    return l2_norm(x)
  if kind == "linf":
    return linf_norm(x)
  raise ValueError(f"Unsupported norm kind {kind!r}; expected 'l2' or 'linf'.")


def all_finite(x: _typing.PyTree) -> jax.Array:
  """Returns True as a scalar boolean JAX array iff all leaves are finite."""
  leaves = jax.tree.leaves(x)
  if not leaves:
    return jnp.asarray(True)
  checks = [jnp.all(jnp.isfinite(leaf)) for leaf in leaves]
  acc = checks[0]
  for check in checks[1:]:
    acc = jnp.logical_and(acc, check)
  return acc


def clip_box(
    x: _typing.PyTree,
    lower: _typing.PyTree,
    upper: _typing.PyTree,
) -> _typing.PyTree:
  """Clips `x` onto `[lower, upper]`, broadcasting scalar bounds."""
  low_scalar = (
      not isinstance(lower, dict | list | tuple) and jnp.ndim(lower) == 0
  )
  high_scalar = (
      not isinstance(upper, dict | list | tuple) and jnp.ndim(upper) == 0
  )
  if low_scalar and high_scalar:
    return jax.tree.map(lambda leaf: jnp.clip(leaf, lower, upper), x)
  if low_scalar:
    return jax.tree.map(lambda leaf, hi: jnp.clip(leaf, lower, hi), x, upper)
  if high_scalar:
    return jax.tree.map(lambda leaf, lo: jnp.clip(leaf, lo, upper), x, lower)
  return jax.tree.map(jnp.clip, x, lower, upper)
