"""Norm-based proximal regularizers (`L1`, `L2`, `NuclearNorm`, etc.)."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import optax

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


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class NuclearNorm:
  """Matrix nuclear (trace) norm `h(X) = weight * ||sigma(X)||_1`."""

  weight: _typing.Scalar = 1.0

  @property
  def strong_convexity(self) -> float:
    """Strong convexity constant (`0.0`)."""
    return 0.0

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates `weight * sum_i sigma_i(X)` across matrix leaves."""
    leaves = jax.tree.leaves(x)
    if not leaves:
      return jnp.asarray(0.0)
    total = tree.functools_sum(
        [jnp.sum(jnp.linalg.svdvals(leaf)) for leaf in leaves]
    )
    return self.weight * total

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Applies singular-value soft-thresholding with `step_size * weight`."""
    thresh = step_size * self.weight

    def _svd_thresh(mat: jax.Array) -> jax.Array:
      u_mat, s_vec, vt_mat = jnp.linalg.svd(mat, full_matrices=False)
      s_shrunk = jnp.maximum(s_vec - thresh, 0.0)
      return (u_mat * s_shrunk[..., None, :]) @ vt_mat

    return jax.tree.map(_svd_thresh, x)

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Identity projection (`dom(NuclearNorm)` is the full matrix space)."""
    return x


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class L21:
  """Group Lasso norm `h(x) = weight * sum_g ||x_g||_2` along `axis`."""

  weight: _typing.Scalar = 1.0
  axis: int = dataclasses.field(default=-1, metadata={"static": True})

  @property
  def strong_convexity(self) -> float:
    """Strong convexity constant (`0.0`)."""
    return 0.0

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates `weight * sum ||x_g||_2` across all leaves."""
    leaves = jax.tree.leaves(x)
    if not leaves:
      return jnp.asarray(0.0)
    total = tree.functools_sum([
        jnp.sum(jnp.linalg.norm(leaf, axis=self.axis))
        for leaf in leaves
    ])
    return self.weight * total

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Applies groupwise block soft-thresholding along `axis`."""
    thresh = step_size * self.weight

    def _group_prox(leaf: jax.Array) -> jax.Array:
      norms = jnp.linalg.norm(leaf, axis=self.axis, keepdims=True)
      safe = jnp.where(norms > 0.0, norms, 1.0)
      scale = jnp.where(norms > thresh, 1.0 - thresh / safe, 0.0)
      return leaf * scale

    return jax.tree.map(_group_prox, x)

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Identity projection."""
    return x


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class LInf:
  """L-infinity norm `h(x) = weight * ||x||_inf`."""

  weight: _typing.Scalar = 1.0

  @property
  def strong_convexity(self) -> float:
    """Strong convexity constant (`0.0`)."""
    return 0.0

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates `weight * max_i |x_i|` across all leaves."""
    leaves = jax.tree.leaves(x)
    if not leaves:
      return jnp.asarray(0.0)
    max_val = jnp.max(
        jnp.stack([jnp.max(jnp.abs(leaf)) for leaf in leaves])
    )
    return self.weight * max_val

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Computes `prox` via Moreau decomposition onto the L1 ball."""
    proj = optax.projections.projection_l1_ball(x, step_size * self.weight)
    return tree.sub(x, proj)

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Identity projection."""
    return x


def _tv1d_adjoint(p: jax.Array) -> jax.Array:
  """Computes `D^T p` along the last axis where `(Du)_i = u_{i+1} - u_i`."""
  return jnp.concatenate(
      [-p[..., :1], p[..., :-1] - p[..., 1:], p[..., -1:]],
      axis=-1,
  )


def _tv1d_prox_leaf(
    arr: jax.Array,
    lam: _typing.Scalar,
    axis: int,
    max_iters: int,
    tol: float,
) -> jax.Array:
  """Solves 1D TV prox along `axis` via restarted dual FISTA."""
  y = jnp.moveaxis(arr, axis, -1)
  n = y.shape[-1]
  if n <= 1:
    return arr

  dual_shape = (*y.shape[:-1], n - 1)
  p0 = jnp.zeros(dual_shape, dtype=y.dtype)
  u0 = y
  t0 = jnp.asarray(1.0, dtype=y.dtype)
  y_scale = jnp.maximum(jnp.max(jnp.abs(y)), 1.0)
  stop_tol = tol * y_scale

  _TVState = tuple[
      jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
  ]

  def _cond(state: _TVState) -> jax.Array:
    _, _, _, _, diff, it = state
    return (it < max_iters) & (diff > stop_tol)

  def _body(state: _TVState) -> _TVState:
    p_cur, z_cur, u_prev, t_cur, _, it = state
    u_grad = y - _tv1d_adjoint(z_cur)
    grad = -(u_grad[..., 1:] - u_grad[..., :-1])
    p_next = jnp.clip(z_cur - 0.25 * grad, -lam, lam)
    u_next = y - _tv1d_adjoint(p_next)
    diff = jnp.max(jnp.abs(u_next - u_prev))

    # Adaptive gradient-based restart when momentum opposes descent.
    restart = jnp.sum((z_cur - p_next) * (p_next - p_cur)) > 0.0
    t_next = jnp.where(
        restart,
        jnp.asarray(1.0, dtype=y.dtype),
        0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t_cur * t_cur)),
    )
    zero_scalar = jnp.asarray(0.0, dtype=y.dtype)
    beta = jnp.where(restart, zero_scalar, (t_cur - 1.0) / t_next)
    z_next = p_next + beta * (p_next - p_cur)
    return p_next, z_next, u_next, t_next, diff, it + 1

  init_diff = jnp.asarray(jnp.inf, dtype=y.dtype)
  init_it = jnp.asarray(0, dtype=jnp.int32)
  _, _, u_final, _, _, _ = jax.lax.while_loop(
      _cond,
      _body,
      (p0, p0, u0, t0, init_diff, init_it),
  )
  return jnp.moveaxis(u_final, -1, axis)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class TV1D:
  """1D Total Variation norm `h(x) = weight * sum_i |x_{i+1} - x_i|`."""

  weight: _typing.Scalar = 1.0
  axis: int = dataclasses.field(default=-1, metadata={"static": True})
  max_iters: int = dataclasses.field(default=500, metadata={"static": True})
  tol: float = dataclasses.field(default=1e-7, metadata={"static": True})

  @property
  def strong_convexity(self) -> float:
    """Strong convexity constant (`0.0`)."""
    return 0.0

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates `weight * sum_i |x_{i+1} - x_i|` along `axis`."""
    leaves = jax.tree.leaves(x)
    if not leaves:
      return jnp.asarray(0.0)
    total = tree.functools_sum([
        jnp.sum(jnp.abs(jnp.diff(leaf, axis=self.axis)))
        for leaf in leaves
    ])
    return self.weight * total

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Computes the 1D TV proximal operator via restarted dual FISTA."""
    lam = step_size * self.weight
    return jax.tree.map(
        lambda leaf: _tv1d_prox_leaf(
            leaf, lam, self.axis, self.max_iters, self.tol
        ),
        x,
    )

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Identity projection."""
    return x


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class Huber:
  """Weighted Huber penalty `weight * sum_i H_delta(x_i)`."""

  weight: _typing.Scalar = 1.0
  delta: _typing.Scalar = 1.0

  @property
  def strong_convexity(self) -> float:
    """Strong convexity constant (`0.0`)."""
    return 0.0

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates the Huber loss across all leaves."""
    leaves = jax.tree.leaves(x)
    if not leaves:
      return jnp.asarray(0.0)
    d = self.delta

    def _leaf_val(u: jax.Array) -> jax.Array:
      abs_u = jnp.abs(u)
      quad = 0.5 * u * u
      lin = d * (abs_u - 0.5 * d)
      return jnp.sum(jnp.where(abs_u <= d, quad, lin))

    return self.weight * tree.functools_sum([_leaf_val(u) for u in leaves])

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Applies the exact elementwise Huber proximal operator."""
    gw = step_size * self.weight
    thresh = self.delta * (1.0 + gw)

    def _prox_leaf(u: jax.Array) -> jax.Array:
      quad_branch = u / (1.0 + gw)
      lin_branch = u - gw * self.delta * jnp.sign(u)
      return jnp.where(jnp.abs(u) <= thresh, quad_branch, lin_branch)

    return jax.tree.map(_prox_leaf, x)

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Identity projection (`dom(Huber)` is the full space)."""
    return x
