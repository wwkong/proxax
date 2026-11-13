"""Min-max saddle-point problem and oracle abstractions."""

from __future__ import annotations

import collections.abc
import dataclasses

import jax

from proxax import _typing
from proxax.prox import base as prox_base


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class MinMaxOracle:
  """First-order oracle for `min_x max_y Phi(x, y) + h(x) - g(y)`."""

  coupling: collections.abc.Callable[
      [_typing.PyTree, _typing.PyTree], jax.Array
  ]
  prox_x: prox_base.ProxOperator = dataclasses.field(
      default_factory=prox_base.Zero
  )
  prox_y: prox_base.ProxOperator = dataclasses.field(
      default_factory=prox_base.Zero
  )
  L_xx: float | None = None
  L_xy: float | None = None
  L_yy: float | None = None
  m_x: float | None = None
  m_y: float | None = None

  def phi_val(self, x: _typing.PyTree, y: _typing.PyTree) -> jax.Array:
    """Evaluates the smooth coupling objective `Phi(x, y)`."""
    return self.coupling(x, y)

  def phi_val_grad(
      self,
      x: _typing.PyTree,
      y: _typing.PyTree,
  ) -> tuple[jax.Array, _typing.PyTree, _typing.PyTree]:
    """Evaluates `(Phi(x, y), grad_x Phi(x, y), grad_y Phi(x, y))`."""
    vag = jax.value_and_grad(self.coupling, argnums=(0, 1))
    val, (gx, gy) = vag(x, y)
    return val, gx, gy

  def grad_x(self, x: _typing.PyTree, y: _typing.PyTree) -> _typing.PyTree:
    """Evaluates `grad_x Phi(x, y)`."""
    return jax.grad(self.coupling, argnums=0)(x, y)

  def grad_y(self, x: _typing.PyTree, y: _typing.PyTree) -> _typing.PyTree:
    """Evaluates `grad_y Phi(x, y)`."""
    return jax.grad(self.coupling, argnums=1)(x, y)

  def obj_val(self, x: _typing.PyTree, y: _typing.PyTree) -> jax.Array:
    """Evaluates the composite saddle objective `Phi(x, y) + h(x) - g(y)`."""
    return self.coupling(x, y) + self.prox_x.value(x) - self.prox_y.value(y)

  def domain_project(
      self,
      x: _typing.PyTree,
      y: _typing.PyTree,
  ) -> tuple[_typing.PyTree, _typing.PyTree]:
    """Projects `(x, y)` onto `dom(h) x dom(g)`."""
    return self.prox_x.domain_project(x), self.prox_y.domain_project(y)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class MinMaxProblem:
  """Immutable problem `min_x max_y Phi(x, y) + h(x) - g(y)`.

  Attributes:
    coupling: Smooth differentiable coupling objective `Phi(x, y)`.
    prox_x: Convex `ProxOperator` `h(x)` (defaults to `prox.Zero()`).
    prox_y: Convex `ProxOperator` `g(y)` (defaults to `prox.Zero()`).
    L_xx: Optional upper Lipschitz constant of `grad_x Phi(., y)`.
    L_xy: Optional cross Lipschitz constant of `(grad_x, grad_y) Phi`.
    L_yy: Optional upper Lipschitz constant of `grad_y Phi(x, .)`.
    m_x: Optional weak-convexity modulus of `Phi(., y)`.
    m_y: Optional weak-concavity modulus of `Phi(x, .)`.
  """

  coupling: collections.abc.Callable[
      [_typing.PyTree, _typing.PyTree], jax.Array
  ]
  prox_x: prox_base.ProxOperator = dataclasses.field(
      default_factory=prox_base.Zero
  )
  prox_y: prox_base.ProxOperator = dataclasses.field(
      default_factory=prox_base.Zero
  )
  L_xx: float | None = None
  L_xy: float | None = None
  L_yy: float | None = None
  m_x: float | None = None
  m_y: float | None = None

  @property
  def oracle(self) -> MinMaxOracle:
    """Returns the `MinMaxOracle` for this problem."""
    return MinMaxOracle(
        coupling=self.coupling,
        prox_x=self.prox_x,
        prox_y=self.prox_y,
        L_xx=self.L_xx,
        L_xy=self.L_xy,
        L_yy=self.L_yy,
        m_x=self.m_x,
        m_y=self.m_y,
    )
