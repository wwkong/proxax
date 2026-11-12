from __future__ import annotations

import jax.numpy as jnp

import proxax
from proxax import prox
from proxax import solvers


def test_ecg_adaptive_and_constant_nonconvex_qp() -> None:
  q_mat = jnp.array([[2.0, -0.5], [-0.5, -1.0]])
  c_vec = jnp.array([-0.4, 0.3])

  prob = proxax.CompositeProblem(
      smooth=proxax.oracle.SmoothTerm(
          fn=lambda x: 0.5 * jnp.dot(x, q_mat @ x) + jnp.dot(c_vec, x),
          upper_curvature=2.5,
          lower_curvature=1.5,
      ),
      proximal=prox.Box(lower=-1.0, upper=1.0),
  )
  x0 = jnp.array([0.8, -0.7])

  sol_adap = proxax.solve(prob, x0, method="ecg", opt_tol=1e-4)
  assert sol_adap.status_enum == proxax.SolveStatus.CONVERGED
  assert float(sol_adap.stats.opt_residual) <= 1e-4

  sol_const = proxax.solve(
      prob,
      x0,
      solver=solvers.ECG(opt_tol=1e-4, step_type="constant"),
  )
  assert sol_const.status_enum == proxax.SolveStatus.CONVERGED
  kkt = prob.evaluate_residuals(sol_const.x, sol_const.subgrad)
  assert float(kkt.stationarity_norm) <= 1e-4
