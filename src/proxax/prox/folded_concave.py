"""Turnkey decompositions for folded-concave nonconvex regularizers.

Each helper splits a folded-concave penalty ``p(x) = q(x) + weight * ||x||_1``
into a differentiable (weakly convex) ``SmoothTerm`` ``q`` and a convex
``L1`` proximal operator ``weight * ||x||_1``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from proxax import _typing
from proxax import tree
from proxax.oracle import terms
from proxax.prox import norms


def _sum_leaves_fn(
    x: _typing.PyTree,
    elem_fn: jax.typing.ArrayLike | object,
) -> jax.Array:
  """Helper to evaluate an elementwise scalar function summed over PyTree."""
  leaves = jax.tree.leaves(x)
  if not leaves:
    return jnp.asarray(0.0)
  return tree.functools_sum([jnp.sum(elem_fn(leaf)) for leaf in leaves])  # type: ignore[operator]


def mcp_decomposition(
    weight: float,
    gamma: float = 3.0,
) -> tuple[terms.SmoothTerm, norms.L1]:
  """Decomposes the Minimax Concave Penalty (MCP) into smooth + L1.

  For scalar ``t``, ``MCP(t) = q(t) + weight * |t|`` where:
  ``q(t) = -t^2 / (2 * gamma)`` for ``|t| <= gamma * weight`` and
  ``q(t) = 0.5 * gamma * weight^2 - weight * |t|`` otherwise.

  Args:
    weight: Nonnegative regularization weight ``lambda``.
    gamma: Concavity parameter ``gamma > 0`` (curvature ``m = 1 / gamma``).

  Returns:
    Tuple ``(smooth_term, l1_prox)`` with exact ``SmoothTerm`` curvatures.
  """
  if weight < 0.0:
    raise ValueError(f"weight must be >= 0, got {weight}.")
  if gamma <= 0.0:
    raise ValueError(f"gamma must be > 0, got {gamma}.")

  thresh = gamma * weight
  curv = 1.0 / gamma

  def _q_elem(t: jax.Array) -> jax.Array:
    abs_t = jnp.abs(t)
    inner = -0.5 * (t * t) / gamma
    outer = 0.5 * gamma * (weight * weight) - weight * abs_t
    return jnp.where(abs_t <= thresh, inner, outer)

  def _q_grad_elem(t: jax.Array) -> jax.Array:
    abs_t = jnp.abs(t)
    return jnp.where(abs_t <= thresh, -t / gamma, -weight * jnp.sign(t))

  def _fn(x: _typing.PyTree) -> jax.Array:
    return _sum_leaves_fn(x, _q_elem)

  def _vag(x: _typing.PyTree) -> tuple[jax.Array, _typing.PyTree]:
    val = _fn(x)
    grad = jax.tree.map(_q_grad_elem, x)
    return val, grad

  smooth = terms.SmoothTerm(
      _fn,
      vag_fn=_vag,
      upper_curvature=curv,
      lower_curvature=curv,
  )
  return smooth, norms.L1(weight=weight)


def scad_decomposition(
    weight: float,
    a: float = 3.7,
) -> tuple[terms.SmoothTerm, norms.L1]:
  """Decomposes the Smoothly Clipped Absolute Deviation (SCAD) penalty.

  For scalar ``t``, ``SCAD(t) = q(t) + weight * |t|`` where ``q`` is ``C^1``
  with weak-convexity constant ``m = 1 / (a - 1)``.

  Args:
    weight: Nonnegative regularization weight ``lambda``.
    a: Shape parameter ``a > 2`` (default ``3.7``).

  Returns:
    Tuple ``(smooth_term, l1_prox)`` with exact ``SmoothTerm`` curvatures.
  """
  if weight < 0.0:
    raise ValueError(f"weight must be >= 0, got {weight}.")
  if a <= 2.0:
    raise ValueError(f"a must be > 2, got {a}.")

  denom = a - 1.0
  curv = 1.0 / denom
  upper = a * weight

  def _q_elem(t: jax.Array) -> jax.Array:
    abs_t = jnp.abs(t)
    mid = -0.5 * ((abs_t - weight) ** 2) / denom
    outer = 0.5 * (a + 1.0) * (weight * weight) - weight * abs_t
    return jnp.where(
        abs_t <= weight,
        jnp.zeros_like(t),
        jnp.where(abs_t <= upper, mid, outer),
    )

  def _q_grad_elem(t: jax.Array) -> jax.Array:
    abs_t = jnp.abs(t)
    sgn = jnp.sign(t)
    mid_g = -sgn * (abs_t - weight) / denom
    outer_g = -weight * sgn
    return jnp.where(
        abs_t <= weight,
        jnp.zeros_like(t),
        jnp.where(abs_t <= upper, mid_g, outer_g),
    )

  def _fn(x: _typing.PyTree) -> jax.Array:
    return _sum_leaves_fn(x, _q_elem)

  def _vag(x: _typing.PyTree) -> tuple[jax.Array, _typing.PyTree]:
    val = _fn(x)
    grad = jax.tree.map(_q_grad_elem, x)
    return val, grad

  smooth = terms.SmoothTerm(
      _fn,
      vag_fn=_vag,
      upper_curvature=curv,
      lower_curvature=curv,
  )
  return smooth, norms.L1(weight=weight)


def logsum_decomposition(
    weight: float,
    eps: float = 0.1,
) -> tuple[terms.SmoothTerm, norms.L1]:
  """Decomposes the Log-Sum penalty ``weight * eps * log(1 + |t| / eps)``.

  Normalized so its slope at ``0^+`` equals ``weight``, making
  ``q(t) = weight * eps * log(1 + |t| / eps) - weight * |t|`` continuously
  differentiable with ``q'(0) = 0`` and curvature ``m = weight / eps``.

  Args:
    weight: Nonnegative regularization weight ``lambda``.
    eps: Scale parameter ``eps > 0``.

  Returns:
    Tuple ``(smooth_term, l1_prox)`` with exact ``SmoothTerm`` curvatures.
  """
  if weight < 0.0:
    raise ValueError(f"weight must be >= 0, got {weight}.")
  if eps <= 0.0:
    raise ValueError(f"eps must be > 0, got {eps}.")

  curv = weight / eps

  def _q_elem(t: jax.Array) -> jax.Array:
    abs_t = jnp.abs(t)
    return weight * eps * jnp.log1p(abs_t / eps) - weight * abs_t

  def _q_grad_elem(t: jax.Array) -> jax.Array:
    abs_t = jnp.abs(t)
    return -weight * t / (eps + abs_t)

  def _fn(x: _typing.PyTree) -> jax.Array:
    return _sum_leaves_fn(x, _q_elem)

  def _vag(x: _typing.PyTree) -> tuple[jax.Array, _typing.PyTree]:
    val = _fn(x)
    grad = jax.tree.map(_q_grad_elem, x)
    return val, grad

  smooth = terms.SmoothTerm(
      _fn,
      vag_fn=_vag,
      upper_curvature=curv,
      lower_curvature=curv,
  )
  return smooth, norms.L1(weight=weight)


def capped_l1_decomposition(
    weight: float,
    theta: float = 1.0,
    smoothing: float = 0.0,
) -> tuple[terms.SmoothTerm, norms.L1]:
  """Decomposes Capped-L1 ``weight * min(|t|, theta)`` into ``q + L1``.

  When ``smoothing == 0.0``, ``q(t) = -weight * max(|t| - theta, 0)`` is
  piecewise linear (nonsmooth at ``|t| = theta``, curvatures ``None``).
  When ``smoothing > 0.0``, ``max(|t| - theta, 0)`` is Huber-smoothed over
  width ``smoothing``, yielding a ``C^1`` term with curvature
  ``weight / smoothing``.

  Args:
    weight: Nonnegative regularization weight ``lambda``.
    theta: Cap threshold ``theta > 0``.
    smoothing: Optional Huber smoothing width ``>= 0``.

  Returns:
    Tuple ``(smooth_term, l1_prox)``.
  """
  if weight < 0.0:
    raise ValueError(f"weight must be >= 0, got {weight}.")
  if theta <= 0.0:
    raise ValueError(f"theta must be > 0, got {theta}.")
  if smoothing < 0.0:
    raise ValueError(f"smoothing must be >= 0, got {smoothing}.")

  def _q_elem(t: jax.Array) -> jax.Array:
    excess = jnp.maximum(jnp.abs(t) - theta, 0.0)
    if smoothing > 0.0:
      quad = 0.5 * (excess * excess) / smoothing
      lin = excess - 0.5 * smoothing
      return -weight * jnp.where(excess <= smoothing, quad, lin)
    return -weight * excess

  def _q_grad_elem(t: jax.Array) -> jax.Array:
    abs_t = jnp.abs(t)
    excess = jnp.maximum(abs_t - theta, 0.0)
    sgn = jnp.sign(t)
    if smoothing > 0.0:
      frac = jnp.minimum(excess / smoothing, 1.0)
      return -weight * sgn * frac
    return jnp.where(abs_t > theta, -weight * sgn, jnp.zeros_like(t))

  def _fn(x: _typing.PyTree) -> jax.Array:
    return _sum_leaves_fn(x, _q_elem)

  def _vag(x: _typing.PyTree) -> tuple[jax.Array, _typing.PyTree]:
    val = _fn(x)
    grad = jax.tree.map(_q_grad_elem, x)
    return val, grad

  curv = (weight / smoothing) if smoothing > 0.0 else None
  smooth = terms.SmoothTerm(
      _fn,
      vag_fn=_vag,
      upper_curvature=curv,
      lower_curvature=curv,
  )
  return smooth, norms.L1(weight=weight)
