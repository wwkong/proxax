from __future__ import annotations

import jax.numpy as jnp

import proxax
from proxax import prox


def test_nc_fista_nonconvex_and_convex() -> None:
  q_mat = jnp.array([[2.0, -0.4], [-0.4, -0.8]])
  c_vec = jnp.array([-0.5, 0.3])

  prob_ncvx = proxax.CompositeProblem(
      smooth=proxax.oracle.SmoothTerm(
          fn=lambda x: 0.5 * jnp.dot(x, q_mat @ x) + jnp.dot(c_vec, x),
          upper_curvature=2.5,
          lower_curvature=1.0,
      ),
      proximal=prox.Box(lower=-1.0, upper=1.0),
  )
  sol_ncvx = proxax.solve(
      prob_ncvx,
      jnp.array([0.8, -0.6]),
      method="nc_fista",
      opt_tol=1e-4,
  )
  assert sol_ncvx.status_enum == proxax.SolveStatus.CONVERGED
  assert float(sol_ncvx.stats.opt_residual) <= 1e-4

  prob_cvx = proxax.CompositeProblem(
      smooth=proxax.oracle.SmoothTerm(
          fn=lambda x: 0.5 * jnp.sum((x - jnp.array([0.4, -0.3])) ** 2),
          upper_curvature=1.0,
          lower_curvature=0.0,
      ),
      proximal=prox.L1(0.05),
  )
  sol_cvx = proxax.solve(
      prob_cvx,
      jnp.array([1.0, -1.0]),
      method="nc_fista",
      opt_tol=1e-4,
  )
  assert sol_cvx.status_enum == proxax.SolveStatus.CONVERGED
  kkt = prob_cvx.evaluate_residuals(sol_cvx.x, sol_cvx.subgrad)
  assert float(kkt.stationarity_norm) <= 1e-4
