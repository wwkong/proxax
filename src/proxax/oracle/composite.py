"""First-order composite oracles and PyTree-parameterized transformations."""

from __future__ import annotations

import dataclasses
import typing

import jax
import jax.numpy as jnp

from proxax import _typing
from proxax import tree
from proxax.constraints import linear_op
from proxax.oracle import terms
from proxax.prox import base as prox_base


@typing.runtime_checkable
class CompositeOracleProtocol(typing.Protocol):
  """Structural protocol for a composite objective `phi(x) = f(x) + h(x)`."""

  @property
  def curvature_upper(self) -> jax.Array | float | None:
    """Optional global smoothness bound `L^+` of `f`."""
    ...

  @property
  def curvature_lower(self) -> jax.Array | float | None:
    """Optional weak-convexity bound `m = L^-` of `f`."""
    ...

  @property
  def strong_convexity(self) -> jax.Array | float:
    """Known strong convexity constant `mu >= 0` of `h`."""
    ...

  def smooth_value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates the smooth component `f(x)`."""
    ...

  def smooth_value_and_grad(
      self,
      x: _typing.PyTree,
  ) -> tuple[jax.Array, _typing.PyTree]:
    """Evaluates `(f(x), grad_f(x))`."""
    ...

  def proximal_value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates the proximal component `h(x)`."""
    ...

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates total composite objective `f(x) + h(x)`."""
    ...

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Evaluates `prox_{step_size * h}(x)`."""
    ...

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Projects `x` onto `cl(dom(h))`."""
    ...


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class CompositeOracle:
  """Composite oracle wrapping a `smooth` term and a `proximal` operator."""

  smooth: terms.SmoothLike = dataclasses.field(default_factory=terms.ZeroSmooth)
  proximal: prox_base.ProxOperator = dataclasses.field(
      default_factory=prox_base.Zero
  )

  def __post_init__(self) -> None:
    """Coerces callable or `None` smooth terms into `SmoothObjective`."""
    if self.smooth is None or (
        callable(self.smooth)
        and not isinstance(self.smooth, terms.SmoothObjective)
    ):
      object.__setattr__(self, "smooth", terms.as_smooth_term(self.smooth))

  @property
  def _smooth_obj(self) -> terms.SmoothObjective:
    return typing.cast(terms.SmoothObjective, self.smooth)

  @property
  def curvature_upper(self) -> jax.Array | float | None:
    """Optional smoothness bound `L^+` of `smooth`."""
    return self._smooth_obj.curvature_upper

  @property
  def curvature_lower(self) -> jax.Array | float | None:
    """Optional weak-convexity bound `m = L^-` of `smooth`."""
    return self._smooth_obj.curvature_lower

  @property
  def strong_convexity(self) -> jax.Array | float:
    """Strong convexity constant `mu >= 0` of `proximal`."""
    return self.proximal.strong_convexity

  def smooth_value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates `smooth.value(x)`."""
    return self._smooth_obj.value(x)

  def smooth_value_and_grad(
      self,
      x: _typing.PyTree,
  ) -> tuple[jax.Array, _typing.PyTree]:
    """Evaluates `smooth.value_and_grad(x)`."""
    return self._smooth_obj.value_and_grad(x)

  def proximal_value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates `proximal.value(x)`."""
    return self.proximal.value(x)

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates `smooth(x) + proximal(x)`."""
    return self._smooth_obj.value(x) + self.proximal.value(x)

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Evaluates `proximal.prox(x, step_size)`."""
    return self.proximal.prox(x, step_size)

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Projects `x` onto `cl(dom(proximal))`."""
    return self.proximal.domain_project(x)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ProxRegularizedOracle:
  """Parameterized proximal subproblem oracle with zero closure re-tracing.

  Represents:
    `f_sub(x) = f(x) + (smooth_reg / 2) ||x - center||_2^2`
    `h_sub(x) = h(x) + (prox_reg / 2) ||x - center||_2^2`

  Keeping `smooth_reg` and `prox_reg` separate ensures we never overestimate
  strong convexity `mu` when `base.curvature_lower` is unknown.
  """

  base: CompositeOracleProtocol
  center: _typing.PyTree
  smooth_reg: jax.Array
  prox_reg: jax.Array

  @property
  def curvature_upper(self) -> jax.Array | float | None:
    """Updated smoothness bound `L^+ + smooth_reg` if `L^+` is known."""
    if self.base.curvature_upper is None:
      return None
    return self.base.curvature_upper + self.smooth_reg

  @property
  def curvature_lower(self) -> jax.Array | float | None:
    """Updated weak convexity `max(0, m - smooth_reg)` if `m` is known."""
    if self.base.curvature_lower is None:
      return None
    return jnp.maximum(0.0, self.base.curvature_lower - self.smooth_reg)

  @property
  def strong_convexity(self) -> jax.Array:
    """Subproblem strong convexity `base.strong_convexity + prox_reg`."""
    return (
        jnp.asarray(self.base.strong_convexity, dtype=self.prox_reg.dtype)
        + self.prox_reg
    )

  def smooth_value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates `f(x) + 0.5 * smooth_reg * ||x - center||_2^2`."""
    diff = tree.sub(x, self.center)
    reg_val = 0.5 * self.smooth_reg * tree.norm_sq(diff)
    return self.base.smooth_value(x) + reg_val

  def smooth_value_and_grad(
      self,
      x: _typing.PyTree,
  ) -> tuple[jax.Array, _typing.PyTree]:
    """Evaluates `(f_sub(x), grad_f(x) + smooth_reg * (x - center))`."""
    val, grad = self.base.smooth_value_and_grad(x)
    diff = tree.sub(x, self.center)
    sub_val = val + 0.5 * self.smooth_reg * tree.norm_sq(diff)
    sub_grad = tree.add_scaled(grad, self.smooth_reg, diff)
    return sub_val, sub_grad

  def proximal_value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates `h(x) + 0.5 * prox_reg * ||x - center||_2^2`."""
    diff_sq = tree.norm_sq(tree.sub(x, self.center))
    return self.base.proximal_value(x) + 0.5 * self.prox_reg * diff_sq

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates `f_sub(x) + h_sub(x)`."""
    return self.smooth_value(x) + self.proximal_value(x)

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Evaluates exact shifted `prox_{step_size * h_sub}(x)`."""
    tau = step_size * self.prox_reg
    denom = 1.0 + tau
    shifted = jax.tree.map(
        lambda u, z: (u + tau * z) / denom,
        x,
        self.center,
    )
    return self.base.prox(shifted, step_size / denom)

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Projects `x` onto `cl(dom(h))`."""
    return self.base.domain_project(x)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class RescaledProxOracle:
  """Parameterized rescaled proximal subproblem oracle for `PF.APD` (`APD`).

  Represents Eq. (3.3) / (3.12) of Kong (SIAM J. Optim. 2024):

  - `psi^s(z) = scale * f(z) + 0.5 * ||z - center||_2^2`
  - `psi^n(z) = scale * h(z)`

  where `scale = 1 / (2 * m_hat) > 0`. Placing the unit quadratic regularization
  in `psi^s` while scaling `h` by `scale` matches `Oracle.proxify` in
  `wwkong/nc_opt` without closure re-tracing under `jax.jit` or `jax.vmap`.
  """

  base: CompositeOracleProtocol
  center: _typing.PyTree
  scale: jax.Array

  @property
  def curvature_upper(self) -> jax.Array | float | None:
    """Upper smoothness bound `scale * L^+ + 1` if `L^+` is known."""
    if self.base.curvature_upper is None:
      return None
    return self.scale * self.base.curvature_upper + 1.0

  @property
  def curvature_lower(self) -> jax.Array | float | None:
    """Weak-convexity bound `max(0, scale * m - 1)` if `m` is known."""
    if self.base.curvature_lower is None:
      return None
    return jnp.maximum(0.0, self.scale * self.base.curvature_lower - 1.0)

  @property
  def strong_convexity(self) -> jax.Array:
    """Strong convexity `scale * base.strong_convexity` of `psi^n`."""
    return self.scale * jnp.asarray(
        self.base.strong_convexity,
        dtype=self.scale.dtype,
    )

  def smooth_value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates `scale * f(x) + 0.5 * ||x - center||_2^2`."""
    diff = tree.sub(x, self.center)
    return self.scale * self.base.smooth_value(x) + 0.5 * tree.norm_sq(diff)

  def smooth_value_and_grad(
      self,
      x: _typing.PyTree,
  ) -> tuple[jax.Array, _typing.PyTree]:
    """Evaluates `(psi^s(x), scale * grad_f(x) + (x - center))`."""
    val, grad = self.base.smooth_value_and_grad(x)
    diff = tree.sub(x, self.center)
    sub_val = self.scale * val + 0.5 * tree.norm_sq(diff)
    sub_grad = tree.add_scaled(diff, self.scale, grad)
    return sub_val, sub_grad

  def proximal_value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates `scale * h(x)`."""
    return self.scale * self.base.proximal_value(x)

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates `psi^s(x) + psi^n(x)`."""
    return self.smooth_value(x) + self.proximal_value(x)

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Evaluates `prox_{(step_size * scale) * h}(x)`."""
    return self.base.prox(x, self.scale * step_size)

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Projects `x` onto `cl(dom(h))`."""
    return self.base.domain_project(x)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class PenalizedOracle:
  """Parameterized penalty / Augmented Lagrangian composite oracle.

  Augments `base`'s smooth component with the Moreau-envelope penalty
  `psi_c(x; p_bar) = (c / 2) ||w(x) - Pi_C(w(x))||_2^2 - ||p_bar||_2^2 / (2c)`,
  where `w(x) = A(x) + p_bar / c` and `p_bar = (1 - dampen) * multiplier`.
  Because `psi_c` is convex, `curvature_lower` of `base` is preserved.
  """

  base: CompositeOracleProtocol
  constraint: linear_op.BoundConstraint
  multiplier: _typing.PyTree | None
  penalty: jax.Array
  dampen: jax.Array | float = 0.0

  @property
  def curvature_upper(self) -> jax.Array | float | None:
    """Upper smoothness `L^+ + penalty * ||A||_op^2` if both are known."""
    op_norm_sq = self.constraint.operator_norm_sq
    if self.base.curvature_upper is None or op_norm_sq is None:
      return None
    return self.base.curvature_upper + self.penalty * op_norm_sq

  @property
  def curvature_lower(self) -> jax.Array | float | None:
    """Weak-convexity bound `m = L^-` (unchanged by convex penalty!)."""
    return self.base.curvature_lower

  @property
  def strong_convexity(self) -> jax.Array | float:
    """Strong convexity constant `mu >= 0` of `base`."""
    return self.base.strong_convexity

  def smooth_value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates `f(x) + psi_c(x; p_bar)`."""
    psi, _ = self.constraint.penalty_and_dual(
        x,
        multiplier=self.multiplier,
        penalty=self.penalty,
        dampen=self.dampen,
    )
    return self.base.smooth_value(x) + psi

  def smooth_value_and_grad(
      self,
      x: _typing.PyTree,
  ) -> tuple[jax.Array, _typing.PyTree]:
    """Evaluates `(f(x) + psi_c(x), grad_f(x) + A^*(p_hat(x)))`."""
    f_val, f_grad = self.base.smooth_value_and_grad(x)
    psi, p_hat = self.constraint.penalty_and_dual(
        x,
        multiplier=self.multiplier,
        penalty=self.penalty,
        dampen=self.dampen,
    )
    pen_grad = self.constraint.adjoint(p_hat)
    return f_val + psi, tree.add(f_grad, pen_grad)

  def effective_dual(self, x: _typing.PyTree) -> _typing.PyTree:
    """Returns `p_hat(x) = c * (w(x) - Pi_C(w(x)))` certifying normal cone."""
    _, p_hat = self.constraint.penalty_and_dual(
        x,
        multiplier=self.multiplier,
        penalty=self.penalty,
        dampen=self.dampen,
    )
    return p_hat

  def proximal_value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates `base.proximal_value(x)`."""
    return self.base.proximal_value(x)

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates `f(x) + psi_c(x) + h(x)`."""
    return self.smooth_value(x) + self.proximal_value(x)

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Evaluates `base.prox(x, step_size)`."""
    return self.base.prox(x, step_size)

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Projects `x` onto `cl(dom(h))`."""
    return self.base.domain_project(x)
