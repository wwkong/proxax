"""Nonconvex spectral matrix completion (`nonconvex_spectral_mc.m`).

Ports `examples/spectral/nonconvex_spectral_mc.m` and
`utils/generators/test_fn_spectral_mc_01.m` from `wwkong/nc_opt` (Kong &
Monteiro, *arXiv:2007.11772*, 2020):

  min_{X in B_R}  0.5 * ||P_Omega(X - M_obs)||_F^2
                  + alpha * beta * (1 - exp(-||sigma(X)||_2^2 / (2 * theta)))
                  + mu * sum_i beta * log(1 + sigma_i(X) / theta)

Using the classic DC / weakly-convex splitting with `kappa_0 = beta / theta`:
- `f_1(X) = 0.5 * ||P_Omega(X - M_obs)||_F^2` (`M_1 = 1, m_1 = 0`),
- `f_2^V(s) = alpha * kappa_tilde(s) + mu * sum_i (kappa(s_i) - kappa_0 |s_i|)`
  (smooth weakly-convex singular-value term),
- `h^V(s) = (mu * kappa_0) * ||s||_1 + delta_{||s||_2 <= R}(s)` (convex
  l1-plus-Euclidean-ball singular-value regularizer).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import optax

import proxax
from proxax import _typing
from proxax import oracle
from proxax import prox
from proxax import solvers
from proxax import tree
from proxax.oracle import terms


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class BallL1VectorProx:
  """Singular-value prox for `weight * ||s||_1 + delta_{||s||_2 <= radius}`."""

  weight: float
  radius: float

  @property
  def strong_convexity(self) -> float:
    """Strong convexity constant (`0.0`)."""
    return 0.0

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates `weight * ||s||_1` at feasible singular-value vectors."""
    s_vec = jnp.asarray(x)
    return self.weight * jnp.sum(jnp.abs(s_vec))

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Applies soft-thresholding followed by Euclidean ball projection."""
    s_vec = jnp.asarray(x)
    thresh = step_size * self.weight
    shrunk = jnp.sign(s_vec) * jnp.maximum(jnp.abs(s_vec) - thresh, 0.0)
    return optax.projections.projection_l2_ball(shrunk, self.radius)

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Projects onto `||s||_2 <= radius`."""
    s_vec = jnp.asarray(x)
    return optax.projections.projection_l2_ball(s_vec, self.radius)


def run_spectral_matrix_completion() -> None:
  """Compares `IAICG`, `DAICG`, and general `APD` on spectral completion."""
  print("=== Nonconvex Log-Sum Spectral Matrix Completion (IA-ICG/DA-ICG) ===")
  key = jax.random.key(777)
  k_u, k_v, k_mask, k_noise, k_init = jax.random.split(key, 5)
  rows, cols, true_rank = 18, 15, 3

  u_true = jax.random.normal(k_u, (rows, true_rank))
  v_true = jax.random.normal(k_v, (cols, true_rank))
  mat_true = u_true @ v_true.T
  mask = (jax.random.uniform(k_mask, (rows, cols)) < 0.65).astype(jnp.float32)
  noise = 0.05 * jax.random.normal(k_noise, (rows, cols))
  mat_obs = mask * (mat_true + noise)

  alpha = 0.25
  beta = 2.0
  theta = 1.0
  mu_reg = 0.25
  kappa0 = beta / theta
  rho = beta / (theta**2)
  radius = float(jnp.sqrt(rows * cols) * jnp.max(jnp.abs(mat_obs)))

  m1_upper, m1_lower = 1.0, 0.0
  m2_upper = alpha * beta / theta
  m2_lower = 2.0 * rho * mu_reg + alpha * (2.0 * beta / theta) * float(
      jnp.exp(-1.5 * theta)
  )

  def f1_fn(mat: jax.Array) -> jax.Array:
    diff = mask * (mat - mat_obs)
    return 0.5 * jnp.sum(diff * diff)

  def f2_vec_fn(s_vec: jax.Array) -> jax.Array:
    abs_s = jnp.abs(s_vec)
    kappa_s = beta * jnp.log(1.0 + abs_s / theta)
    kappa_tilde = beta * (
        1.0 - jnp.exp(-jnp.sum(s_vec * s_vec) / (2.0 * theta))
    )
    return alpha * kappa_tilde + mu_reg * jnp.sum(kappa_s - kappa0 * abs_s)

  def ball_proj(mat: jax.Array) -> jax.Array:
    frob = tree.norm(mat)
    scale = jnp.minimum(1.0, radius / jnp.maximum(frob, 1e-12))
    return mat * scale

  spec_problem = oracle.SpectralProblem(
      f1=terms.SmoothTerm(
          f1_fn, upper_curvature=m1_upper, lower_curvature=m1_lower
      ),
      f2_vec=terms.SmoothTerm(
          f2_vec_fn, upper_curvature=m2_upper, lower_curvature=m2_lower
      ),
      prox_vec=BallL1VectorProx(weight=mu_reg * kappa0, radius=radius),
      domain_proj=ball_proj,
  )

  x0 = 0.1 * jax.random.normal(k_init, (rows, cols))
  tol = 1e-3

  spec_suite: list[
      tuple[str, solvers.IAICG | solvers.DAICG | solvers.Solver]
  ] = [
      ("IAICG (constant)", solvers.IAICG(opt_tol=tol, step_type="constant")),
      ("IAICG (adaptive)", solvers.IAICG(opt_tol=tol, step_type="adaptive")),
      ("DAICG (monotone)", solvers.DAICG(opt_tol=tol, monotone=True)),
      ("APD (matrix SVD)", solvers.APD(opt_tol=tol)),
  ]
  for name, solver in spec_suite:
    sol = proxax.solve(spec_problem, x0, solver=solver)
    s_vals = jnp.linalg.svdvals(sol.x)
    num_nz = int(jnp.sum(s_vals > 1e-2))
    rel_err = float(
        jnp.linalg.norm(sol.x - mat_true) / jnp.linalg.norm(mat_true)
    )
    print(
        f"{name:<18} status={sol.status_enum.name:<9} "
        f"obj={float(sol.stats.obj_val):.4f} "
        f"||V_hat||_F={float(sol.stats.opt_residual):.2e} "
        f"rank={num_nz}/{min(rows, cols)} rel_err={rel_err:.3f} "
        f"outer={int(sol.stats.outer_iters)}"
    )

  print("\n=== Convex Nuclear-Norm Baseline via CompositeProblem ===")
  nuc_problem = proxax.CompositeProblem(
      smooth=terms.SmoothTerm(f1_fn, upper_curvature=1.0, lower_curvature=0.0),
      proximal=prox.NuclearNorm(weight=mu_reg * kappa0),
  )
  sol_nuc = proxax.solve(nuc_problem, x0, solver=solvers.DAICG(opt_tol=tol))
  rank_nuc = int(jnp.sum(jnp.linalg.svdvals(sol_nuc.x) > 1e-2))
  err_nuc = float(
      jnp.linalg.norm(sol_nuc.x - mat_true) / jnp.linalg.norm(mat_true)
  )
  print(
      f"NuclearNorm DAICG  status={sol_nuc.status_enum.name:<9} "
      f"obj={float(sol_nuc.stats.obj_val):.4f} "
      f"||V_hat||_F={float(sol_nuc.stats.opt_residual):.2e} "
      f"rank={rank_nuc}/{min(rows, cols)} rel_err={err_nuc:.3f}"
  )


if __name__ == "__main__":
  run_spectral_matrix_completion()
