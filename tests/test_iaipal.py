from __future__ import annotations

import jax.numpy as jnp

import proxax
from proxax import constraints
from proxax import prox
from proxax import solvers


def test_iaipal_equality_and_inequality_constrained_nonconvex_qp() -> None:
  q_mat = jnp.array([[2.0, -0.4], [-0.4, -0.8]])
  c_vec = jnp.array([-0.3, 0.5])
  a_mat = jnp.array([[1.0, 1.0]])
  b_vec = jnp.array([0.4])

  prob = proxax.CompositeProblem(
      smooth=lambda x: 0.5 * jnp.dot(x, q_mat @ x) + jnp.dot(c_vec, x),
      proximal=prox.Box(lower=-1.0, upper=1.0),
      constraints=constraints.equality(a_mat, b_vec),
  )
  sol = proxax.solve(
      prob,
      jnp.array([0.1, 0.1]),
      method="iaipal",
      opt_tol=1e-3,
      feas_tol=1e-3,
  )
  assert sol.status_enum == proxax.SolveStatus.CONVERGED
  assert sol.dual is not None
  assert sol.subgrad is not None
  kkt = prob.evaluate_residuals(sol.x, sol.subgrad)
  assert float(kkt.primal_residual_norm) <= 1e-3
  assert float(kkt.stationarity_norm) <= 1e-3

  sol_var = proxax.solve(
      prob,
      jnp.array([0.1, 0.1]),
      solver=solvers.IAIPAL(
          opt_tol=1e-3,
          feas_tol=1e-3,
          lower_curvature=1.0,
          upper_curvature=2.5,
          sigma_type="variable",
          k0_type=2,
      ),
  )
  assert sol_var.status_enum == proxax.SolveStatus.CONVERGED
  assert float(sol_var.stats.opt_residual) <= 1e-3
  assert float(sol_var.stats.feas_residual) <= 1e-3
