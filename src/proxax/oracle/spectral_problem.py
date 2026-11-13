"""Spectral problem and orthogonally invariant prox definitions."""

from __future__ import annotations

import dataclasses
import typing

import jax
import jax.numpy as jnp

from proxax import _typing
from proxax import problem as problem_mod
from proxax import tree
from proxax.oracle import spectral
from proxax.oracle import terms
from proxax.prox import base as prox_base
from proxax.prox import norms as prox_norms


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class SpectralProx:
  """Orthogonally invariant matrix prox `(prox_vec o sigma)(X)`."""

  prox_vec: prox_base.ProxOperator

  @property
  def strong_convexity(self) -> jax.Array | float:
    """Strong convexity constant `mu >= 0` of `prox_vec`."""
    return self.prox_vec.strong_convexity

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates `prox_vec(sigma(X))`."""
    mat = typing.cast(jax.Array, x)
    if isinstance(self.prox_vec, prox_base.Zero):
      return jnp.asarray(0.0, dtype=mat.dtype)
    return self.prox_vec.value(jnp.linalg.svdvals(mat))

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Evaluates `P diag(prox_{step_size * prox_vec}(s)) Q^T`."""
    mat = typing.cast(jax.Array, x)
    if isinstance(self.prox_vec, prox_base.Zero):
      return mat
    p_mat, s_vec, qt_mat = jnp.linalg.svd(mat, full_matrices=False)
    s_p = typing.cast(jax.Array, self.prox_vec.prox(s_vec, step_size))
    return spectral.reconstruct(p_mat, s_p, qt_mat)

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Projects singular values of `X` onto `cl(dom(prox_vec))`."""
    mat = typing.cast(jax.Array, x)
    if isinstance(self.prox_vec, prox_base.Zero):
      return mat
    p_mat, s_vec, qt_mat = jnp.linalg.svd(mat, full_matrices=False)
    s_d = typing.cast(jax.Array, self.prox_vec.domain_project(s_vec))
    return spectral.reconstruct(p_mat, s_d, qt_mat)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class SpectralProblem:
  """Unconstrained spectral problem `f1(X) + (f2_vec + h)(sigma(X))`.

  Attributes:
    f1: Smooth matrix objective `f_1(X)`.
    f2_vec: Smooth absolutely symmetric singular-value objective `f_2^V(s)`.
    prox_vec: Convex symmetric singular-value `ProxOperator` `h^V(s)`.
    domain_proj: Optional matrix projection `Pi_Omega(X)` for `DA-ICG`.
  """

  f1: terms.SmoothLike = dataclasses.field(default_factory=terms.ZeroSmooth)
  f2_vec: terms.SmoothLike = dataclasses.field(default_factory=terms.ZeroSmooth)
  prox_vec: prox_base.ProxOperator = dataclasses.field(
      default_factory=prox_base.Zero
  )
  domain_proj: typing.Callable[[jax.Array], jax.Array] | None = None

  @property
  def is_constrained(self) -> bool:
    """Returns False (`SpectralProblem` is unconstrained over matrices)."""
    return False

  @property
  def oracle(self) -> spectral.SpectralOracle:
    """Returns the underlying `SpectralOracle`."""
    return spectral.SpectralOracle(
        f1=self.f1,
        f2_vec=self.f2_vec,
        prox_vec=self.prox_vec,
        domain_proj=self.domain_proj,
    )

  def as_composite_problem(self) -> problem_mod.CompositeProblem:
    """Converts this `SpectralProblem` into a general `CompositeProblem`."""
    spec_o = self.oracle
    smooth_term = terms.SmoothTerm(
        spec_o.smooth_value,
        vag_fn=spec_o.smooth_value_and_grad,
        upper_curvature=spec_o.curvature_upper,
        lower_curvature=spec_o.curvature_lower,
    )
    return problem_mod.CompositeProblem(
        smooth=smooth_term,
        proximal=SpectralProx(self.prox_vec),
    )

  def evaluate_residuals(
      self,
      x: jax.Array,
      subgrad: jax.Array | None,
      *,
      norm_kind: str = "l2",
  ) -> problem_mod.KKTResiduals:
    """Evaluates objective value and stationarity residual norm."""
    obj = self.oracle.value(x)
    stat_norm = (
        tree.norm(subgrad, kind=norm_kind)
        if subgrad is not None
        else jnp.asarray(jnp.inf, dtype=x.dtype)
    )
    return problem_mod.KKTResiduals(
        primal_residual_norm=jnp.asarray(0.0, dtype=x.dtype),
        stationarity_norm=stat_norm,
        objective_value=obj,
    )


jax.tree_util.register_dataclass(
    SpectralProblem,
    data_fields=["f1", "f2_vec", "prox_vec"],
    meta_fields=["domain_proj"],
)


_COMPOSITE_PROX_MAP: dict[
    type[prox_base.ProxOperator],
    typing.Callable[[typing.Any], prox_base.ProxOperator],
] = {
    prox_norms.NuclearNorm: lambda p: prox_norms.L1(weight=p.weight),
    prox_norms.L2Squared: lambda p: prox_norms.L2Squared(weight=p.weight),
    prox_norms.L2: lambda p: prox_norms.L2(weight=p.weight),
}


def as_spectral_oracle(
    problem_or_oracle: (
        SpectralProblem | spectral.SpectralOracle | problem_mod.CompositeProblem
    ),
) -> spectral.SpectralOracle:
  """Extracts or lifts a `SpectralOracle` from a spectral or nuclear problem."""
  if isinstance(problem_or_oracle, spectral.SpectralOracle):
    return problem_or_oracle
  if isinstance(problem_or_oracle, SpectralProblem):
    return problem_or_oracle.oracle
  if isinstance(problem_or_oracle, problem_mod.CompositeProblem):
    if problem_or_oracle.is_constrained:
      raise ValueError(
          "Spectral ICG solvers solve unconstrained matrix problems."
      )
    prox_op = problem_or_oracle.proximal
    if isinstance(prox_op, prox_base.Zero):
      return spectral.SpectralOracle(f1=problem_or_oracle.smooth)
    prox_builder = _COMPOSITE_PROX_MAP.get(type(prox_op))
    if prox_builder is not None:
      return spectral.SpectralOracle(
          f1=problem_or_oracle.smooth,
          prox_vec=prox_builder(prox_op),
      )
    raise ValueError(
        "Pass `SpectralProblem(f1=..., f2_vec=..., prox_vec=...)` or a "
        "`CompositeProblem` with `NuclearNorm`, `L2`, `L2Squared`, or `Zero`."
    )
  raise TypeError(
      "Unsupported problem type for spectral solver: "
      f"{type(problem_or_oracle)!r}."
  )
