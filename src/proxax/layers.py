"""Drop-in differentiable optimization layers built on `px.implicit_solve`."""

from __future__ import annotations

import dataclasses
import typing

import jax
import jax.numpy as jnp

from proxax import _typing
from proxax import implicit
from proxax import problem as problem_mod
from proxax import prox as prox_mod
from proxax.constraints import linear_op
from proxax.oracle import terms


@dataclasses.dataclass(frozen=True, slots=True)
class ProxLayer:
  """Differentiable proximal projection layer `argmin_x 0.5*||x - s||^2 + h(x)`.

  Works directly inside `jax.grad`, `jax.jit`, `jax.vmap`, Optax, Flax, and
  Equinox training loops via implicit differentiation (`px.implicit_solve`).
  """

  proximal: typing.Any
  method: str = "panoc"
  opt_tol: float = 1e-5
  linear_tol: float = 1e-5
  linear_solver: str = "gmres"
  damping: float = 1e-6
  reg: float = 0.0

  def _make_problem(
      self, scores: _typing.PyTree
  ) -> problem_mod.CompositeProblem:
    reg = self.reg

    def _smooth(x: _typing.PyTree) -> jax.Array:
      diff_sq = sum(
          (
              jnp.sum((lx - ls) ** 2)
              for lx, ls in zip(
                  jax.tree.leaves(x), jax.tree.leaves(scores), strict=True
              )
          ),
          jnp.asarray(0.0),
      )
      if reg == 0.0:
        return 0.5 * diff_sq
      reg_sq = sum(
          (jnp.sum(lx**2) for lx in jax.tree.leaves(x)),
          jnp.asarray(0.0),
      )
      return 0.5 * diff_sq + 0.5 * reg * reg_sq

    return problem_mod.CompositeProblem(
        smooth=terms.as_smooth_term(_smooth),
        proximal=self.proximal,
    )

  def __call__(
      self,
      scores: _typing.PyTree,
      x0: _typing.PyTree | None = None,
  ) -> _typing.PyTree:
    """Computes `argmin_x 0.5 * ||x - scores||^2 + h(x)` with `custom_vjp`."""
    init = (
        jax.tree.map(jnp.zeros_like, scores)
        if x0 is None
        else x0
    )
    return implicit.implicit_solve(
        self._make_problem,
        init,
        scores,
        method=self.method,
        opt_tol=self.opt_tol,
        linear_tol=self.linear_tol,
        linear_solver=self.linear_solver,
        damping=self.damping,
    )

  def batched(
      self,
      scores_batch: _typing.PyTree,
      x0_batch: _typing.PyTree | None = None,
  ) -> _typing.PyTree:
    """Vectorized batch evaluation via `jax.vmap`."""
    if x0_batch is None:
      return jax.vmap(lambda s: self(s))(scores_batch)
    return jax.vmap(lambda s, x0: self(s, x0))(scores_batch, x0_batch)


@dataclasses.dataclass(frozen=True, slots=True)
class CappedSimplexLayer(ProxLayer):
  """Differentiable capped-simplex layer `{0 <= x <= upper, sum(x) = scale}`."""

  scale: float = 1.0
  upper: float = 1.0
  proximal: typing.Any = dataclasses.field(init=False)

  def __post_init__(self) -> None:
    """Initializes the `CappedSimplex` proximal operator."""
    object.__setattr__(
        self,
        "proximal",
        prox_mod.CappedSimplex(scale=self.scale, upper=self.upper),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class SimplexLayer(ProxLayer):
  """Differentiable simplex (sparsemax) layer `{x >= 0, sum(x) = scale}`."""

  scale: float = 1.0
  proximal: typing.Any = dataclasses.field(init=False)

  def __post_init__(self) -> None:
    """Initializes the `Simplex` proximal operator."""
    object.__setattr__(self, "proximal", prox_mod.Simplex(scale=self.scale))


@dataclasses.dataclass(frozen=True, slots=True)
class BoxLayer(ProxLayer):
  """Differentiable box-constrained projection layer `{lower <= x <= upper}`."""

  lower: _typing.PyTree = 0.0
  upper: _typing.PyTree = 1.0
  proximal: typing.Any = dataclasses.field(init=False)

  def __post_init__(self) -> None:
    """Initializes the `Box` proximal operator."""
    object.__setattr__(
        self,
        "proximal",
        prox_mod.Box(lower=self.lower, upper=self.upper),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class QPLayer:
  """Differentiable equality-constrained QP layer via `panoc_alm`.

  Solves `argmin_x 0.5 * x^T Q x + <c, x> + h(x)  s.t.  A x = b` with
  implicit differentiation with respect to `(c, A, b)`.
  """

  Q: jax.Array
  proximal: typing.Any = dataclasses.field(default_factory=prox_mod.Zero)
  method: str = "panoc_alm"
  opt_tol: float = 1e-5
  feas_tol: float = 1e-5
  linear_tol: float = 1e-5
  linear_solver: str = "gmres"
  damping: float = 1e-6

  def __call__(
      self,
      c: jax.Array,
      A: jax.Array,
      b: jax.Array,
      x0: jax.Array | None = None,
  ) -> jax.Array:
    """Solves the parameterized QP and differentiates w.r.t. `(c, A, b)`."""
    Q = self.Q
    prox_op = self.proximal

    def _prob(
        params: tuple[jax.Array, jax.Array, jax.Array],
    ) -> problem_mod.CompositeProblem:
      c_p, A_p, b_p = params

      def _quad(x: jax.Array) -> jax.Array:
        return 0.5 * jnp.dot(x, Q @ x) + jnp.dot(c_p, x)

      return problem_mod.CompositeProblem(
          smooth=terms.as_smooth_term(_quad),
          proximal=prox_op,
          constraints=linear_op.equality(A_p, b_p),
      )

    init = jnp.zeros_like(c) if x0 is None else x0
    return typing.cast(
        jax.Array,
        implicit.implicit_solve(
            _prob,
            init,
            (c, A, b),
            method=self.method,
            opt_tol=self.opt_tol,
            feas_tol=self.feas_tol,
            linear_tol=self.linear_tol,
            linear_solver=self.linear_solver,
            damping=self.damping,
        ),
    )


__all__ = [
    "BoxLayer",
    "CappedSimplexLayer",
    "ProxLayer",
    "QPLayer",
    "SimplexLayer",
]
