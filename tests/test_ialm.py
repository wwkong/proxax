from __future__ import annotations

import jax.numpy as jnp

import proxax
from proxax import constraints
from proxax import prox
from proxax import solvers


def test_ialm_apd_equality_constrained_nonconvex_qp() -> None:
  q_mat = jnp.array([[2.0, -0.4], [-0.4, -0.8]])
  c_vec = jnp.array([-0.3, 0.5])
  a_mat = jnp.array([[1.0, 1.0]])
  b_vec = jnp.array([0.4])

  prob = proxax.CompositeProblem(
      smooth=lambda x: 0.5 * jnp.dot(x, q_mat @ x) + jnp.dot(c_vec, x),
      proximal=prox.Box(lower=-1.0, upper=1.0),
      constraints=constraints.equality(a_mat, b_vec),
  )
  solver = solvers.IALM(
      opt_tol=1e-3,
      feas_tol=1e-3,
      subproblem_solver="apd",
      penalty_init=2.0,
      penalty_factor=3.0,
      feas_decay=0.5,
      max_alm_iters=12,
      max_subproblem_iters=25,
      max_inner_iters=50,
  )
  fin = solver.solve_state(prob, jnp.array([0.1, 0.1]))

  assert int(fin.status) == int(proxax.SolveStatus.CONVERGED)
  assert float(fin.best_opt_res) <= 1e-3
  assert float(fin.best_feas_res) <= 1e-3
  assert float(fin.penalty) <= 54.0
  kkt = prob.evaluate_residuals(fin.best_x, fin.best_subgrad)
  assert float(kkt.primal_residual_norm) <= 1e-3
  assert float(kkt.stationarity_norm) <= 1e-3


def test_ialm_aidal_dampened_pytree_product_constraints() -> None:
  prob = proxax.CompositeProblem(
      smooth=lambda p: 0.5 * jnp.sum((p["w"] - 0.6) ** 2)
      - 0.2 * jnp.sum(p["b"] ** 2),
      proximal=prox.MultiProx({
          "w": prox.L1(0.05),
          "b": prox.Box(lower=0.0, upper=1.0),
      }),
      constraints={
          "eq": constraints.equality(
              lambda p: p["w"] + p["b"],
              rhs=jnp.array([0.8]),
          ),
          "le": constraints.less_equal(
              lambda p: p["w"] - p["b"],
              rhs=jnp.array([0.3]),
          ),
      },
  )
  solver = solvers.IALM(
      opt_tol=1e-3,
      feas_tol=1e-3,
      subproblem_solver="aipp",
      dampen=0.2,
      dual_step=0.8,
      penalty_init=2.0,
      penalty_factor=3.0,
      max_alm_iters=15,
      max_subproblem_iters=25,
      max_inner_iters=50,
  )
  x0 = {"w": jnp.array([0.3]), "b": jnp.array([0.3])}
  sol = proxax.solve(prob, x0, solver=solver)

  assert sol.status_enum == proxax.SolveStatus.CONVERGED
  assert isinstance(sol.dual, dict)
  assert set(sol.dual.keys()) == {"eq", "le"}
  assert jnp.all(sol.dual["le"] >= -1e-6)
  kkt = prob.evaluate_residuals(sol.x, sol.subgrad)
  assert float(kkt.primal_residual_norm) <= 1e-3
  assert float(kkt.stationarity_norm) <= 1e-3
