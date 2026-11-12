from __future__ import annotations

import jax.numpy as jnp

import proxax
from proxax import prox


def test_adap_fista_nonconvex_box_qp_including_zero_init() -> None:
  q_mat = jnp.array([[2.0, -0.4], [-0.4, -0.8]])
  c_vec = jnp.array([-0.5, 0.3])

  prob = proxax.CompositeProblem(
      smooth=lambda x: 0.5 * jnp.dot(x, q_mat @ x) + jnp.dot(c_vec, x),
      proximal=prox.Box(lower=-1.0, upper=1.0),
  )
  sol = proxax.solve(prob, jnp.zeros(2), method="adap_fista", opt_tol=1e-4)
  assert sol.status_enum == proxax.SolveStatus.CONVERGED
  kkt = prob.evaluate_residuals(sol.x, sol.subgrad)
  assert float(kkt.stationarity_norm) <= 1e-4
