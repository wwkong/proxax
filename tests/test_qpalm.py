from __future__ import annotations

import jax.numpy as jnp
from jax.experimental import sparse as jsparse

import proxax
from proxax import constraints
from proxax import prox
from proxax import solvers


def test_qpalm_dense_box_and_equality() -> None:
  p_mat = jnp.array([[4.0, 1.0], [1.0, 2.0]])
  q_vec = jnp.array([1.0, 1.0])
  a_mat = jnp.array([[1.0, 1.0]])
  b_vec = jnp.array([1.0])

  prob = proxax.CompositeProblem(
      smooth=lambda x: 0.5 * jnp.dot(x, p_mat @ x) + jnp.dot(q_vec, x),
      proximal=prox.Box(lower=0.0, upper=0.7),
      constraints=constraints.equality(a_mat, b_vec),
  )
  solver = solvers.QPALM(opt_tol=1e-6, feas_tol=1e-6, linear_solver="dense")
  fin = solver.solve_state(prob, jnp.zeros(2))

  assert int(fin.status) == int(proxax.SolveStatus.CONVERGED)
  assert float(fin.best_opt_res) <= 1e-6
  assert float(fin.best_feas_res) <= 1e-6
  kkt = prob.evaluate_residuals(fin.best_x, fin.best_subgrad)
  assert float(kkt.primal_residual_norm) <= 1e-6
  assert float(kkt.stationarity_norm) <= 1e-6


def test_qpalm_qdldl_ffi_bcoo_box_inequality() -> None:
  p_dense = jnp.array([
      [3.0, -1.0, 0.0, 0.0],
      [-1.0, 3.0, -1.0, 0.0],
      [0.0, -1.0, 3.0, -1.0],
      [0.0, 0.0, -1.0, 3.0],
  ])
  a_dense = jnp.array([
      [1.0, 1.0, 0.0, 0.0],
      [0.0, 1.0, 1.0, 1.0],
  ])
  p_bcoo = jsparse.BCOO.fromdense(p_dense)
  a_bcoo = jsparse.BCOO.fromdense(a_dense)
  q_vec = jnp.array([-2.0, 1.0, -3.0, 0.5])

  prob = proxax.CompositeProblem(
      smooth=lambda x: 0.5 * jnp.dot(x, p_bcoo @ x) + jnp.dot(q_vec, x),
      proximal=prox.Box(lower=-0.5, upper=1.5),
      constraints=constraints.box(
          a_bcoo,
          lower=jnp.array([-1.0, 0.5]),
          upper=jnp.array([0.8, 2.0]),
      ),
  )
  solver = solvers.QPALM(
      opt_tol=1e-6, feas_tol=1e-6, linear_solver="qdldl_ffi"
  )
  fin = solver.solve_state(prob, jnp.zeros(4))

  assert int(fin.status) == int(proxax.SolveStatus.CONVERGED)
  assert float(fin.best_opt_res) <= 1e-6
  assert float(fin.best_feas_res) <= 1e-6


def test_qpalm_equilibrate_ill_conditioned_qp() -> None:
  scales = jnp.array([1e5, 1.0, 1e-3])
  p_mat = jnp.diag(scales)
  q_vec = jnp.array([-1e5, -2.0, 1e-3])
  a_mat = jnp.array([[1e3, 1.0, 1e-2], [0.0, 1e-2, 1e2]])

  prob = proxax.CompositeProblem(
      smooth=lambda x: 0.5 * jnp.dot(x, p_mat @ x) + jnp.dot(q_vec, x),
      proximal=prox.Box(lower=-2.0, upper=2.0),
      constraints=constraints.box(
          a_mat,
          lower=jnp.array([-jnp.inf, -5.0]),
          upper=jnp.array([500.0, 10.0]),
      ),
  )
  solver = solvers.QPALM(
      opt_tol=1e-6,
      feas_tol=1e-6,
      linear_solver="qdldl_ffi",
      equilibrate=True,
  )
  fin = solver.solve_state(prob, jnp.zeros(3))

  assert int(fin.status) == int(proxax.SolveStatus.CONVERGED)
  assert float(fin.best_opt_res) <= 1e-6
  assert float(fin.best_feas_res) <= 1e-6
  kkt = prob.evaluate_residuals(fin.best_x, fin.best_subgrad)
  assert float(kkt.primal_residual_norm) <= 1e-6
  assert float(kkt.stationarity_norm) <= 1e-6


def test_qpalm_px_solve_method_dispatch() -> None:
  p_mat = jnp.array([[2.0, 0.0], [0.0, 2.0]])
  a_mat = jnp.array([[1.0, 2.0]])
  prob = proxax.CompositeProblem(
      smooth=lambda x: 0.5 * jnp.dot(x, p_mat @ x) - x[0] - x[1],
      proximal=prox.Zero(),
      constraints=constraints.less_equal(a_mat, jnp.array([1.0])),
  )
  sol = proxax.solve(
      prob, jnp.zeros(2), method="qpalm", opt_tol=1e-6, feas_tol=1e-6
  )
  assert int(sol.status) == int(proxax.SolveStatus.CONVERGED)
  assert float(sol.stats.opt_residual) <= 1e-6
  assert float(sol.stats.feas_residual) <= 1e-6
