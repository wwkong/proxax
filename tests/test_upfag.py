from __future__ import annotations

import jax.numpy as jnp

import proxax
from proxax import prox
from proxax import solvers


def test_upfag_monotonic_and_bb_nonconvex_qp() -> None:
  q_mat = jnp.array([[2.0, -0.4], [-0.4, -0.8]])
  c_vec = jnp.array([-0.5, 0.3])

  prob = proxax.CompositeProblem(
      smooth=lambda x: 0.5 * jnp.dot(x, q_mat @ x) + jnp.dot(c_vec, x),
      proximal=prox.Box(lower=-1.0, upper=1.0),
  )
  x0 = jnp.array([0.8, -0.6])

  sol_mono = proxax.solve(prob, x0, method="upfag", opt_tol=1e-3)
  assert sol_mono.status_enum == proxax.SolveStatus.CONVERGED
  assert float(sol_mono.stats.opt_residual) <= 1e-3

  sol_bb = proxax.solve(
      prob,
      x0,
      solver=solvers.UPFAG(opt_tol=1e-3, line_search_type="bb"),
  )
  assert sol_bb.status_enum == proxax.SolveStatus.CONVERGED
  kkt = prob.evaluate_residuals(sol_bb.x, sol_bb.subgrad)
  assert float(kkt.stationarity_norm) <= 1e-3
