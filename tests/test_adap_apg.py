from __future__ import annotations

import jax.numpy as jnp

import proxax
from proxax import prox
from proxax import solvers


def test_adap_apg_strongly_convex_and_nonconvex() -> None:
  a_mat = jnp.array([[3.0, 0.5], [0.5, 1.5]])
  b_vec = jnp.array([-0.7, 0.4])
  sc_prob = proxax.CompositeProblem(
      smooth=lambda x: 0.5 * jnp.dot(x, a_mat @ x) + jnp.dot(b_vec, x),
      proximal=prox.ElasticNet(l1_weight=0.1, l2_weight=0.5),
  )
  sol_sc = proxax.solve(
      sc_prob,
      jnp.array([1.0, -1.0]),
      solver=solvers.AdapAPG(opt_tol=1e-5, monotone=False),
  )
  assert sol_sc.status_enum == proxax.SolveStatus.CONVERGED
  assert sol_sc.subgrad is not None
  _, grad_sc = sc_prob.oracle.smooth_value_and_grad(sol_sc.x)
  fp_sc = sc_prob.oracle.prox(sol_sc.x + sol_sc.subgrad - grad_sc, 1.0)
  assert float(jnp.linalg.norm(sol_sc.x - fp_sc)) <= 1e-5

  q_nc = jnp.array([[2.0, -0.4], [-0.4, -0.8]])
  nc_prob = proxax.CompositeProblem(
      smooth=lambda x: 0.5 * jnp.dot(x, q_nc @ x) + jnp.dot(b_vec, x),
      proximal=prox.Box(lower=-1.0, upper=1.0),
  )
  sol_nc = proxax.solve(nc_prob, jnp.zeros(2), method="adap_apg", opt_tol=1e-4)
  assert sol_nc.status_enum == proxax.SolveStatus.CONVERGED
  kkt_nc = nc_prob.evaluate_residuals(sol_nc.x, sol_nc.subgrad)
  assert float(kkt_nc.stationarity_norm) <= 1e-4
