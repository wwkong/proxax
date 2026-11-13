"""Shared setup helpers for spectral `IAICG` and `DAICG` solvers."""

from __future__ import annotations

import dataclasses
import typing

import jax
import jax.numpy as jnp

from proxax import _typing
from proxax import certificates
from proxax import problem as problem_mod
from proxax import tree
from proxax.oracle import spectral
from proxax.solvers import base
from proxax.solvers.inner import acg


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ICGInit:
  """Common initialization bundle for `IAICG` and `DAICG`."""

  z0_hat: jax.Array
  z0_svals: jax.Array
  phi0: jax.Array
  dtype: typing.Any
  tracker0: base.SolverTracker
  m2_low: jax.Array
  m2_est0: jax.Array
  xi0: jax.Array
  lam0: jax.Array
  acg_cfg: acg.ACGConfig


def init_icg_setup(
    oracle: spectral.SpectralOracle,
    x0: jax.Array,
    *,
    solver_name: str,
    opt_tol: float,
    rel_opt_tol: float,
    opt_tol_override: _typing.Scalar | None,
    norm_kind: str,
    lipschitz_init: float,
    xi_init: float | None,
    lam_init: float | None,
    sigma: float | None,
    max_inner_iters: int,
    jit: bool,
) -> ICGInit:
  """Validates 2D input, runs step-0 refinement, and calibrates xi0 & lam0."""
  z0_proj = typing.cast(
      jax.Array,
      oracle.domain_project(tree.as_float_tree(x0)),
  )
  if z0_proj.ndim != 2:
    raise ValueError(
        f"`{solver_name}` expects a 2D matrix `x0`; got shape {z0_proj.shape}."
    )
  dtype = z0_proj.dtype

  ref0 = certificates.refine(
      oracle,
      z0_proj,
      jnp.asarray(lipschitz_init, dtype=dtype),
      jit=jit,
  )
  z0_hat = typing.cast(jax.Array, ref0.x_hat)
  v0_hat = typing.cast(jax.Array, ref0.v_hat)
  res0 = tree.norm(v0_hat, kind=norm_kind)
  tracker0 = base.SolverTracker.from_init(
      z0_hat,
      v0_hat,
      res0,
      ref0.obj_val,
      opt_tol=opt_tol,
      rel_opt_tol=rel_opt_tol,
      opt_tol_override=opt_tol_override,
      smooth_evals=ref0.smooth_evals,
      grad_evals=ref0.grad_evals,
      prox_evals=ref0.prox_evals,
  )

  known_m1 = oracle.f1_obj.curvature_upper
  known_m2 = oracle.f2_vec_obj.curvature_upper
  known_m2_low = oracle.f2_vec_obj.curvature_lower

  m1_base = (
      jnp.asarray(known_m1, dtype=dtype)
      if known_m1 is not None
      else ref0.lipschitz_used
  )
  m1_base = jnp.maximum(m1_base, jnp.asarray(1e-4, dtype=dtype))
  m2_low = (
      jnp.asarray(known_m2_low, dtype=dtype)
      if known_m2_low is not None
      else jnp.asarray(0.0, dtype=dtype)
  )
  m2_est0 = (
      jnp.asarray(known_m2, dtype=dtype)
      if known_m2 is not None
      else jnp.asarray(1.0, dtype=dtype)
  )

  xi0 = jnp.asarray(xi_init, dtype=dtype) if xi_init is not None else m1_base
  xi0 = jnp.maximum(xi0, jnp.asarray(1e-4, dtype=dtype))

  if lam_init is not None:
    lam0 = jnp.asarray(lam_init, dtype=dtype)
  elif known_m1 is not None:
    lam0 = jnp.where(m1_base <= xi0, 5.0 / m1_base, 0.25 / m1_base)
  else:
    lam0 = 1.0 / m1_base

  sigma_val = sigma if sigma is not None else 0.3
  acg_cfg = acg.ACGConfig(
      sigma=sigma_val,
      max_iters=max_inner_iters,
  )
  return ICGInit(
      z0_hat=z0_hat,
      z0_svals=jnp.linalg.svdvals(z0_hat),
      phi0=ref0.obj_val,
      dtype=dtype,
      tracker0=tracker0,
      m2_low=m2_low,
      m2_est0=m2_est0,
      xi0=xi0,
      lam0=lam0,
      acg_cfg=acg_cfg,
  )


def as_spectral_oracle(
    problem: spectral.SpectralProblem | problem_mod.CompositeProblem,
) -> spectral.SpectralOracle:
  """Converts a `SpectralProblem` or `CompositeProblem` to `SpectralOracle`."""
  if isinstance(problem, spectral.SpectralProblem):
    return problem.oracle
  return spectral.as_spectral_oracle(problem)
