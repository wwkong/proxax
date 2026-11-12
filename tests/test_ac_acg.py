from __future__ import annotations

import jax.numpy as jnp

import proxax
from proxax import prox
from proxax import solvers


def test_ac_acg_nonconvex_box_qp_exact_certificate() -> None:
  q_mat = jnp.array([[2.0, -0.4], [-0.4, -0.8]])
  c_vec = jnp.array([-0.5, 0.3])

  prob = proxax.CompositeProblem(
      smooth=lambda x: 0.5 * jnp.dot(x, q_mat @ x) + jnp.dot(c_vec, x),
      proximal=prox.Box(lower=-1.0, upper=1.0),
  )
  sol = proxax.solve(
      prob,
      jnp.array([0.8, -0.6]),
      solver=solvers.ACACG(opt_tol=1e-4, upper_curvature=2.5),
  )
  assert sol.status_enum == proxax.SolveStatus.CONVERGED
  assert sol.subgrad is not None
  kkt = prob.evaluate_residuals(sol.x, sol.subgrad)
  assert float(kkt.stationarity_norm) <= 1e-4

  _, grad_x = prob.oracle.smooth_value_and_grad(sol.x)
  fixed_pt = prob.oracle.prox(sol.x + sol.subgrad - grad_x, 1.0)
  assert float(jnp.linalg.norm(sol.x - fixed_pt)) <= 1e-5

  sol_auto = proxax.solve(prob, jnp.zeros(2), method="ac_acg", opt_tol=1e-4)
  assert sol_auto.status_enum == proxax.SolveStatus.CONVERGED
  assert float(sol_auto.stats.opt_residual) <= 1e-4
