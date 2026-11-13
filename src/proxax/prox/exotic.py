"""Conic, sorting-based, barrier, and sparse proximal operators."""

from __future__ import annotations

import collections.abc as abc
import dataclasses
import typing

import jax
import jax.numpy as jnp
import optax

from proxax import _typing
from proxax import tree
from proxax.constraints import sets as constraint_sets


def _pava_nondecreasing(y: jax.Array) -> jax.Array:
  """Pool Adjacent Violators Algorithm for non-decreasing isotonic fit."""
  n = y.shape[0]
  if n <= 1:
    return y
  dtype = y.dtype
  vals = jnp.zeros(n, dtype=dtype)
  counts = jnp.zeros(n, dtype=jnp.int32)
  top = jnp.asarray(-1, dtype=jnp.int32)

  def _push(
      i: jax.Array,
      carry: tuple[jax.Array, jax.Array, jax.Array],
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    v_buf, c_buf, t = carry
    t = t + 1
    v_buf = v_buf.at[t].set(y[i])
    c_buf = c_buf.at[t].set(1)

    def _cond(st: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
      vb, _, tp = st
      return (tp > 0) & (vb[tp - 1] > vb[tp])

    def _merge(
        st: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
      vb, cb, tp = st
      c_prev = cb[tp - 1]
      c_curr = cb[tp]
      c_sum = c_prev + c_curr
      merged = (
          vb[tp - 1] * c_prev.astype(dtype) + vb[tp] * c_curr.astype(dtype)
      ) / c_sum.astype(dtype)
      vb = vb.at[tp - 1].set(merged)
      cb = cb.at[tp - 1].set(c_sum)
      return vb, cb, tp - 1

    return jax.lax.while_loop(_cond, _merge, (v_buf, c_buf, t))

  vals, counts, top = jax.lax.fori_loop(0, n, _push, (vals, counts, top))
  ends = jnp.cumsum(counts)
  idx = jnp.arange(n, dtype=jnp.int32)
  # Mask inactive tail slots beyond `top` so searchsorted maps to valid blocks.
  valid = jnp.arange(n, dtype=jnp.int32) <= top
  safe_ends = jnp.where(valid, ends, n + 1)
  block_ids = jnp.searchsorted(safe_ends, idx, side="right")
  return typing.cast(jax.Array, vals[block_ids])


def _pava_nonincreasing(y: jax.Array) -> jax.Array:
  """Pool Adjacent Violators Algorithm for non-increasing isotonic fit."""
  return -_pava_nondecreasing(-y)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class PSDCone:
  """Indicator of the symmetric positive semidefinite cone `X >= 0`."""

  @property
  def strong_convexity(self) -> float:
    """Strong convexity constant (`0.0`)."""
    return 0.0

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Returns `0.0` at feasible points."""
    return jnp.asarray(0.0, dtype=tree.infer_dtype(x))

  def _project(self, x: jax.Array) -> jax.Array:
    sym = 0.5 * (x + jnp.swapaxes(x, -1, -2))
    evals, evecs = jnp.linalg.eigh(sym)
    evals_pos = jnp.maximum(evals, 0.0)
    return typing.cast(
        jax.Array,
        (evecs * evals_pos[..., None, :]) @ jnp.swapaxes(evecs, -1, -2),
    )

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Projects symmetric matrix leaves onto the PSD cone."""
    del step_size
    return jax.tree.map(self._project, x)

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Projects symmetric matrix leaves onto the PSD cone."""
    return jax.tree.map(self._project, x)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class Spectrahedron:
  """Indicator of the spectrahedron `{X >= 0 : trace(X) = scale}`."""

  scale: _typing.Scalar = 1.0

  @property
  def strong_convexity(self) -> float:
    """Strong convexity constant (`0.0`)."""
    return 0.0

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Returns `0.0` at feasible points."""
    return jnp.asarray(0.0, dtype=tree.infer_dtype(x))

  def _project(self, x: jax.Array) -> jax.Array:
    sym = 0.5 * (x + jnp.swapaxes(x, -1, -2))
    evals, evecs = jnp.linalg.eigh(sym)
    s = jnp.asarray(self.scale, dtype=evals.dtype)
    evals_proj = s * optax.projections.projection_simplex(evals / s)
    return typing.cast(
        jax.Array,
        (evecs * evals_proj[..., None, :]) @ jnp.swapaxes(evecs, -1, -2),
    )

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Projects symmetric matrix leaves onto the spectrahedron."""
    del step_size
    return jax.tree.map(self._project, x)

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Projects symmetric matrix leaves onto the spectrahedron."""
    return jax.tree.map(self._project, x)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class SOC:
  """Indicator of the second-order (Lorentz) cone `||x[1:]||_2 <= x[0]`."""

  @property
  def strong_convexity(self) -> float:
    """Strong convexity constant (`0.0`)."""
    return 0.0

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Returns `0.0` at feasible points."""
    return jnp.asarray(0.0, dtype=tree.infer_dtype(x))

  def _project(self, x: jax.Array) -> jax.Array:
    t = x[0]
    v = x[1:]
    norm_v = jnp.linalg.norm(v)
    safe_norm = jnp.where(norm_v > 0.0, norm_v, 1.0)
    mid = 0.5 * (1.0 + t / safe_norm) * jnp.concatenate(
        [jnp.expand_dims(norm_v, 0), v]
    )
    return jnp.where(
        norm_v <= t,
        x,
        jnp.where(norm_v <= -t, jnp.zeros_like(x), mid),
    )

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Projects 1D array leaves onto the second-order cone."""
    del step_size
    return jax.tree.map(self._project, x)

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Projects 1D array leaves onto the second-order cone."""
    return jax.tree.map(self._project, x)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class Isotonic:
  """Indicator of monotone vectors `x_1 <= ... <= x_n` (or non-increasing)."""

  increasing: bool = dataclasses.field(
      default=True, metadata={"static": True}
  )

  @property
  def strong_convexity(self) -> float:
    """Strong convexity constant (`0.0`)."""
    return 0.0

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Returns `0.0` at feasible points."""
    return jnp.asarray(0.0, dtype=tree.infer_dtype(x))

  def _project(self, x: jax.Array) -> jax.Array:
    return (
        _pava_nondecreasing(x)
        if self.increasing
        else _pava_nonincreasing(x)
    )

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Projects 1D array leaves onto the isotonic cone."""
    del step_size
    return jax.tree.map(self._project, x)

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Projects 1D array leaves onto the isotonic cone."""
    return jax.tree.map(self._project, x)


def _owl_value(x: jax.Array, weights: jax.Array) -> jax.Array:
  u = jnp.sort(jnp.abs(x))[::-1]
  return jnp.sum(weights.astype(x.dtype) * u)


def _owl_prox(
    x: jax.Array, weights: jax.Array, step_size: _typing.Scalar
) -> jax.Array:
  u = jnp.abs(x)
  order = jnp.argsort(-u)
  u_sorted = u[order]
  tau = jnp.asarray(step_size, dtype=x.dtype)
  z_sorted = jnp.maximum(
      _pava_nonincreasing(u_sorted - tau * weights.astype(x.dtype)), 0.0
  )
  inv = jnp.empty_like(order).at[order].set(jnp.arange(x.shape[0]))
  return jnp.sign(x) * z_sorted[inv]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class OWL:
  """Ordered Weighted L1 norm `sum_i weights[i] * |x|_{(i)}`."""

  weights: jax.Array | abc.Sequence[float]

  @property
  def strong_convexity(self) -> float:
    """Strong convexity constant (`0.0`)."""
    return 0.0

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates the OWL norm on 1D array `x`."""
    arr = jnp.asarray(x)
    w = jnp.asarray(self.weights, dtype=arr.dtype)
    return _owl_value(arr, w)

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Evaluates the exact OWL prox via PAVA in `O(n log n)`."""
    arr = jnp.asarray(x)
    w = jnp.asarray(self.weights, dtype=arr.dtype)
    return _owl_prox(arr, w, step_size)

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Identity projection (full-domain norm)."""
    return x


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class OSCAR:
  """OSCAR norm `alpha * ||x||_1 + beta * sum_{i<j} max(|x_i|, |x_j|)`."""

  alpha: _typing.Scalar = 1.0
  beta: _typing.Scalar = 0.1

  @property
  def strong_convexity(self) -> float:
    """Strong convexity constant (`0.0`)."""
    return 0.0

  def _weights(self, n: int, dtype: jnp.dtype) -> jax.Array:
    a = jnp.asarray(self.alpha, dtype=dtype)
    b = jnp.asarray(self.beta, dtype=dtype)
    return a + b * jnp.arange(n - 1, -1, -1, dtype=dtype)

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates the OSCAR norm on 1D array `x`."""
    arr = jnp.asarray(x)
    return _owl_value(arr, self._weights(arr.shape[0], arr.dtype))

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Evaluates the exact OSCAR prox via PAVA."""
    arr = jnp.asarray(x)
    return _owl_prox(arr, self._weights(arr.shape[0], arr.dtype), step_size)

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Identity projection (full-domain norm)."""
    return x


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class CappedSimplex:
  """Indicator of the capped simplex `{0 <= x <= upper, sum(x) = scale}`."""

  scale: _typing.Scalar = 1.0
  upper: _typing.Scalar = 1.0

  @property
  def strong_convexity(self) -> float:
    """Strong convexity constant (`0.0`)."""
    return 0.0

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Returns `0.0` at feasible points."""
    return jnp.asarray(0.0, dtype=tree.infer_dtype(x))

  def _project(self, x: jax.Array) -> jax.Array:
    u = jnp.asarray(self.upper, dtype=x.dtype)
    s = jnp.asarray(self.scale, dtype=x.dtype)
    lo = jnp.min(x) - u
    hi = jnp.max(x)

    def _step(
        _: jax.Array, bounds: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
      a, b = bounds
      mid = 0.5 * (a + b)
      total = jnp.sum(jnp.clip(x - mid, 0.0, u))
      return jnp.where(total > s, mid, a), jnp.where(total > s, b, mid)

    lo_f, hi_f = jax.lax.fori_loop(0, 60, _step, (lo, hi))
    nu = 0.5 * (lo_f + hi_f)
    return jnp.clip(x - nu, 0.0, u)

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Projects 1D array leaves onto the capped simplex."""
    del step_size
    return jax.tree.map(self._project, x)

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Projects 1D array leaves onto the capped simplex."""
    return jax.tree.map(self._project, x)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class LogBarrier:
  """Logarithmic barrier `h(x) = -weight * sum(log(x))` on `x > 0`."""

  weight: _typing.Scalar = 1.0

  @property
  def strong_convexity(self) -> float:
    """Strong convexity constant (`0.0` globally on `x > 0`)."""
    return 0.0

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates `-weight * sum(log(x))` across leaves."""
    w = jnp.asarray(self.weight, dtype=tree.infer_dtype(x))
    total = sum(
        (jnp.sum(jnp.log(leaf)) for leaf in jax.tree.leaves(x)),
        jnp.asarray(0.0, dtype=w.dtype),
    )
    return -w * total

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Closed-form `0.5 * (x + sqrt(x^2 + 4 * step_size * weight))`."""
    dtype = tree.infer_dtype(x)
    prod = jnp.asarray(step_size, dtype=dtype) * jnp.asarray(
        self.weight, dtype=dtype
    )
    return jax.tree.map(
        lambda v: 0.5 * (v + jnp.sqrt(v * v + 4.0 * prod)),
        x,
    )

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Projects onto `cl(dom(h)) = {x >= 0}`."""
    return jax.tree.map(lambda v: jnp.maximum(v, 0.0), x)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class NegEntropy:
  """Negative Shannon entropy `h(x) = weight * sum(x * log(x))` on `x >= 0`."""

  weight: _typing.Scalar = 1.0

  @property
  def strong_convexity(self) -> float:
    """Strong convexity constant (`0.0` on unbounded `x >= 0`)."""
    return 0.0

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates `weight * sum(x * log(x))` with `0 * log(0) = 0`."""
    w = jnp.asarray(self.weight, dtype=tree.infer_dtype(x))

    def _leaf_ent(leaf: jax.Array) -> jax.Array:
      safe = jnp.where(leaf > 0.0, leaf, 1.0)
      return jnp.sum(jnp.where(leaf > 0.0, leaf * jnp.log(safe), 0.0))

    total = sum(
        (_leaf_ent(leaf) for leaf in jax.tree.leaves(x)),
        jnp.asarray(0.0, dtype=w.dtype),
    )
    return w * total

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Exact prox via Newton's method on `e^y + y = z` in log-space."""
    dtype = tree.infer_dtype(x)
    lm = jnp.asarray(step_size, dtype=dtype) * jnp.asarray(
        self.weight, dtype=dtype
    )
    log_lm = jnp.log(lm)

    def _prox_leaf(v: jax.Array) -> jax.Array:
      z = v / lm - 1.0 - log_lm
      y0 = jnp.where(z > 1.0, jnp.log(jnp.maximum(z, 1.0)), z)

      def _newton(_: jax.Array, y: jax.Array) -> jax.Array:
        ey = jnp.exp(jnp.minimum(y, 80.0))
        return y - (ey + y - z) / (ey + 1.0)

      y_sol = jax.lax.fori_loop(0, 12, _newton, y0)
      return lm * jnp.exp(y_sol)

    return jax.tree.map(_prox_leaf, x)

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Projects onto `{x >= 0}`."""
    return jax.tree.map(lambda v: jnp.maximum(v, 0.0), x)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class L0:
  """Nonconvex `L0` pseudo-norm `weight * ||x||_0` (hard thresholding)."""

  weight: _typing.Scalar = 1.0

  @property
  def strong_convexity(self) -> float:
    """Strong convexity constant (`0.0`)."""
    return 0.0

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates `weight * count(x != 0)` across leaves."""
    w = jnp.asarray(self.weight, dtype=tree.infer_dtype(x))
    cnt = sum(
        (jnp.sum(leaf != 0.0).astype(w.dtype) for leaf in jax.tree.leaves(x)),
        jnp.asarray(0.0, dtype=w.dtype),
    )
    return w * cnt

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Hard-thresholds entries with `|x| <= sqrt(2 * step_size * weight)`."""
    dtype = tree.infer_dtype(x)
    thresh = jnp.sqrt(
        2.0
        * jnp.asarray(step_size, dtype=dtype)
        * jnp.asarray(self.weight, dtype=dtype)
    )
    return jax.tree.map(
        lambda v: jnp.where(jnp.abs(v) > thresh, v, 0.0),
        x,
    )

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Identity projection."""
    return x


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class CardinalityBall:
  """Indicator of the sparse `L0` ball `{x : ||x||_0 <= k}`."""

  k: int = dataclasses.field(metadata={"static": True})

  @property
  def strong_convexity(self) -> float:
    """Strong convexity constant (`0.0`)."""
    return 0.0

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Returns `0.0` at feasible points."""
    return jnp.asarray(0.0, dtype=tree.infer_dtype(x))

  def _project(self, x: jax.Array) -> jax.Array:
    flat = x.ravel()
    k_eff = min(max(self.k, 0), flat.shape[0])
    if k_eff == 0:
      return jnp.zeros_like(x)
    if k_eff == flat.shape[0]:
      return x
    _, top_idx = jax.lax.top_k(jnp.abs(flat), k_eff)
    out = jnp.zeros_like(flat).at[top_idx].set(flat[top_idx])
    return out.reshape(x.shape)

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Keeps the `k` largest-magnitude entries per leaf."""
    del step_size
    return jax.tree.map(self._project, x)

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Keeps the `k` largest-magnitude entries per leaf."""
    return jax.tree.map(self._project, x)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class ExpCone:
  """Indicator of the 3D primal exponential cone."""

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
    """Projects each 3-vector leaf onto the primal exponential cone."""
    del step_size
    return jax.tree.map(constraint_sets.project_exp_cone, x)

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Projects each 3-vector leaf onto the primal exponential cone."""
    return jax.tree.map(constraint_sets.project_exp_cone, x)


__all__ = [
    "CardinalityBall",
    "CappedSimplex",
    "ExpCone",
    "Isotonic",
    "L0",
    "LogBarrier",
    "NegEntropy",
    "OSCAR",
    "OWL",
    "PSDCone",
    "SOC",
    "Spectrahedron",
]
