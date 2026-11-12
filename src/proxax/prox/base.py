"""Core protocol and calculus wrappers for proximal operators."""

from __future__ import annotations

import dataclasses
import typing

import jax
import jax.numpy as jnp

from proxax import _typing
from proxax import tree


@typing.runtime_checkable
class ProxOperator(typing.Protocol):
  """Protocol for proper, closed, (strongly) convex proximable functions `h`."""

  @property
  def strong_convexity(self) -> jax.Array | float:
    """Known Euclidean strong convexity constant `mu >= 0` of `h`."""
    ...

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates `h(x)` at a domain point `x`."""
    ...

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Computes `argmin_u { h(u) + (1 / (2 * step_size)) ||u - x||_2^2 }`."""
    ...

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Projects `x` onto `cl(dom(h))`."""
    ...


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class Zero:
  """Zero proximal function `h(x) = 0`."""

  @property
  def strong_convexity(self) -> float:
    """Strong convexity constant (`0.0`)."""
    return 0.0

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Returns scalar `0.0` matching the dtype of `x`."""
    return jnp.asarray(0.0, dtype=tree.infer_dtype(x))

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Identity proximal mapping."""
    del step_size
    return x

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Identity projection onto the full space."""
    return x


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Shifted:
  """Proximal function shifted by a quadratic `(reg / 2) ||x - center||_2^2`.

  Represents `h_tilde(x) = base(x) + (reg / 2) ||x - center||_2^2` with exact
  closed-form proximal mapping:
  `prox_{alpha * h_tilde}(u) = prox_{(alpha / (1 + alpha * reg)) * base}(`
  `    (u + alpha * reg * center) / (1 + alpha * reg))`.

  Attributes:
    base: Underlying `ProxOperator`.
    center: Proximal center PyTree `z_0`.
    reg: Non-negative quadratic weight `reg >= 0` as a JAX array or float.
  """

  base: ProxOperator
  center: _typing.PyTree
  reg: jax.Array

  @property
  def strong_convexity(self) -> jax.Array:
    """Total strong convexity `base.strong_convexity + reg`."""
    base_mu = jnp.asarray(self.base.strong_convexity, dtype=self.reg.dtype)
    return base_mu + self.reg

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates `base(x) + 0.5 * reg * ||x - center||_2^2`."""
    diff_sq = tree.norm_sq(tree.sub(x, self.center))
    return self.base.value(x) + 0.5 * self.reg * diff_sq

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Evaluates the exact shifted proximal operator."""
    tau = step_size * self.reg
    denom = 1.0 + tau
    shifted_input = jax.tree.map(
        lambda u, z: (u + tau * z) / denom,
        x,
        self.center,
    )
    return self.base.prox(shifted_input, step_size / denom)

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Projects `x` onto `cl(dom(base))`."""
    return self.base.domain_project(x)
