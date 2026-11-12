from __future__ import annotations

import jax.numpy as jnp

import proxax
from proxax import prox
from proxax import solvers


def test_ag_known_and_auto_curvature() -> None:
  q_mat = jnp.array([[2.0, -0.4], [-0.4, -0.8]])
  c_vec = jnp.array([-0.5, 0.3])

  prob = proxax.CompositeProblem(
      smooth=lambda x: 0.5 * jnp.dot(x, q_mat @ x) + jnp.dot(c_vec, x),
      proximal=prox.Box(lower=-1.0, upper=1.0),
  )
  x0 = jnp.array([0.8, -0.6])

  sol_known = proxax.solve(
      prob,
      x0,
      solver=solvers.AG(opt_tol=1e-4, upper_curvature=2.5),
  )
  assert sol_known.status_enum == proxax.SolveStatus.CONVERGED
  kkt = prob.evaluate_residuals(sol_known.x, sol_known.subgrad)
  assert float(kkt.stationarity_norm) <= 1e-4

  sol_auto = proxax.solve(prob, x0, method="ag", opt_tol=1e-4)
  assert sol_auto.status_enum == proxax.SolveStatus.CONVERGED
  assert float(sol_auto.stats.opt_residual) <= 1e-4
