from __future__ import annotations

import jax.numpy as jnp

import proxax
from proxax import constraints
from proxax import prox


def test_s_prox_alm_nonconvex_box_qp_with_linear_constraints() -> None:
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
      method="s_prox_alm",
      opt_tol=1e-4,
      feas_tol=1e-4,
  )
  assert sol.status_enum == proxax.SolveStatus.CONVERGED
  assert sol.dual is not None
  assert sol.subgrad is not None
  kkt = prob.evaluate_residuals(sol.x, sol.subgrad)
  assert float(kkt.primal_residual_norm) <= 1e-4
  assert float(kkt.stationarity_norm) <= 1e-4

  bound_c = prob.bind_constraints(sol.x)
  assert bound_c is not None
  _, grad_f = prob.oracle.smooth_value_and_grad(sol.x)
  grad_lag = grad_f + bound_c.adjoint(sol.dual)
  fp_x = prob.oracle.prox(sol.x + sol.subgrad - grad_lag, 1.0)
  assert float(jnp.linalg.norm(sol.x - fp_x)) <= 1e-5
