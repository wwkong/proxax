"""Tests for Modified Ruiz Equilibration (`px.precondition`)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsparse

from proxax import px


def test_ruiz_equilibrate_dense_qp_matches_original_solution() -> None:
  """`solve_preconditioned` solves an ill-scaled QP to original tolerances."""
  scale = jnp.array([1.0, 1e3, 1e-2, 50.0])
  p_mat = jnp.diag(jnp.array([2.0, 2e6, 2e-4, 5000.0]))
  q_vec = jnp.array([-2.0, -2e3, 1e-2, -100.0])
  a_mat = jnp.array(
      [[1.0, 1e3, 0.0, 10.0], [0.0, 100.0, 1e-2, 5.0]], dtype=jnp.float32
  )
  b_vec = jnp.array([2.0, 1.0], dtype=jnp.float32)

  prob = px.CompositeProblem(
      smooth=lambda x: 0.5 * jnp.dot(x, p_mat @ x) + jnp.dot(q_vec, x),
      proximal=px.prox.Box(
          lower=jnp.zeros(4), upper=10.0 / jnp.maximum(scale, 1.0)
      ),
      constraints=px.constraints.less_equal(a_mat, rhs=b_vec),
  )
  x0 = jnp.zeros(4)
  sol = px.solve_preconditioned(
      prob,
      x0,
      method="ialm",
      opt_tol=1e-4,
      feas_tol=1e-4,
      hessian=p_mat,
  )
  assert sol.status_enum == px.SolveStatus.CONVERGED
  bound_c = prob.bind_constraints(sol.x)
  assert bound_c is not None
  feas_orig = float(jnp.linalg.norm(bound_c.residual(sol.x)))
  assert feas_orig <= 1e-4


def test_ruiz_equilibrate_sparse_bcoo_matches_dense() -> None:
  """`ruiz_equilibrate` supports sparse `BCOO` without densifying."""
  p_dense = jnp.diag(jnp.array([4.0, 400.0, 0.04, 16.0]))
  a_dense = jnp.array([[1.0, 100.0, 0.0, 0.0], [0.0, 0.0, 0.1, 4.0]])
  p_bcoo = jsparse.BCOO.fromdense(p_dense)
  a_bcoo = jsparse.BCOO.fromdense(a_dense)
  q_vec = jnp.array([-1.0, -50.0, -0.1, -4.0])
  b_vec = jnp.array([1.0, 1.0])
  x0 = jnp.zeros(4)

  prob_dense = px.CompositeProblem(
      smooth=lambda x: 0.5 * jnp.dot(x, p_dense @ x) + jnp.dot(q_vec, x),
      proximal=px.prox.NonNegative(),
      constraints=px.constraints.equality(a_dense, rhs=b_vec),
  )
  prob_sparse = px.CompositeProblem(
      smooth=lambda x: 0.5 * jnp.dot(x, p_bcoo @ x) + jnp.dot(q_vec, x),
      proximal=px.prox.NonNegative(),
      constraints=px.constraints.equality(a_bcoo, rhs=b_vec),
  )

  pre_d = px.ruiz_equilibrate(prob_dense, x0, hessian=p_dense)
  pre_s = px.ruiz_equilibrate(prob_sparse, x0, hessian=p_bcoo)
  assert isinstance(pre_s.problem.constraints.operator.matrix, jsparse.BCOO)  # type: ignore[union-attr]
  assert jnp.allclose(pre_d.d_x, pre_s.d_x, atol=1e-5)
  assert pre_d.d_y is not None
  assert pre_s.d_y is not None
  assert jnp.allclose(pre_d.d_y, pre_s.d_y, atol=1e-5)
  assert jnp.allclose(pre_d.cost_scale, pre_s.cost_scale, atol=1e-5)

  sol_d = px.solve_preconditioned(
      prob_dense,
      x0,
      method="ialm",
      opt_tol=1e-5,
      feas_tol=1e-5,
      hessian=p_dense,
  )
  sol_s = px.solve_preconditioned(
      prob_sparse,
      x0,
      method="ialm",
      opt_tol=1e-5,
      feas_tol=1e-5,
      hessian=p_bcoo,
  )
  assert sol_d.status_enum == px.SolveStatus.CONVERGED
  assert sol_s.status_enum == px.SolveStatus.CONVERGED
  # In float32, BCOO vs dense matvec rounding can shift IALM's stopping point
  # within opt_tol=1e-5, amplified by 1 / min(diag(P)) = 25 on x[2].
  x_atol = 1e-4 if jax.config.jax_enable_x64 else 1e-2
  assert jnp.allclose(sol_d.x, sol_s.x, atol=x_atol)
  assert jnp.allclose(sol_d.stats.obj_val, sol_s.stats.obj_val, atol=1e-4)


def test_ruiz_equilibrate_l1_prox() -> None:
  """`solve_preconditioned` preserves optimal solution with `px.prox.L1`."""
  a_mat = jnp.array([[10.0, 0.1, 0.0], [0.0, 0.2, 5.0]])
  b_vec = jnp.array([1.0, 1.0])
  prob = px.CompositeProblem(
      smooth=lambda x: 0.5 * jnp.sum((x - jnp.array([1.0, 2.0, -1.0])) ** 2),
      proximal=px.prox.L1(weight=0.25),
      constraints=px.constraints.less_equal(a_mat, rhs=b_vec),
  )
  x0 = jnp.zeros(3)
  sol_ref = px.solve(prob, x0, method="ialm", opt_tol=1e-5, feas_tol=1e-5)
  sol_pre = px.solve_preconditioned(
      prob, x0, method="ialm", opt_tol=1e-5, feas_tol=1e-5
  )
  assert sol_ref.status_enum == px.SolveStatus.CONVERGED
  assert sol_pre.status_enum == px.SolveStatus.CONVERGED
  assert jnp.allclose(sol_ref.x, sol_pre.x, atol=1e-4)


def test_ruiz_equilibrate_product_constraint() -> None:
  """`solve_preconditioned` supports `ProductConstraint` (`dict`) inputs."""
  a_eq = jsparse.BCOO.fromdense(jnp.array([[100.0, 0.1, 1.0]]))
  b_eq = jnp.array([1.0])
  a_ineq = jnp.array([[0.01, 50.0, 0.0], [1.0, 0.0, 200.0]])
  b_ineq = jnp.array([2.0, 3.0])

  prob = px.CompositeProblem(
      smooth=lambda x: 0.5 * jnp.sum((x - jnp.array([1.0, 2.0, -1.0])) ** 2),
      proximal=px.prox.NonNegative(),
      constraints={
          "eq": px.constraints.equality(a_eq, rhs=b_eq),
          "ineq": px.constraints.less_equal(a_ineq, rhs=b_ineq),
      },
  )
  x0 = jnp.zeros(3)
  sol_ref = px.solve(prob, x0, method="ialm", opt_tol=1e-5, feas_tol=1e-5)
  sol_pre = px.solve_preconditioned(
      prob, x0, method="ialm", opt_tol=1e-5, feas_tol=1e-5
  )
  assert sol_ref.status_enum == px.SolveStatus.CONVERGED
  assert sol_pre.status_enum == px.SolveStatus.CONVERGED
  assert jnp.allclose(sol_ref.x, sol_pre.x, atol=1e-4)
  assert sol_ref.dual is not None and sol_pre.dual is not None
  assert jnp.allclose(sol_ref.dual["eq"], sol_pre.dual["eq"], atol=1e-3)
  assert jnp.allclose(sol_ref.dual["ineq"], sol_pre.dual["ineq"], atol=1e-3)


def test_batch_solve_precondition() -> None:
  """`batch_solve(..., precondition=True)` works across `vmap` and `threads`."""
  a_mat = jnp.array([[100.0, 0.01, 1.0], [0.05, 50.0, 2.0]])
  rhs_batch = jnp.array([[1.0, 1.0], [2.0, 0.5], [0.5, 2.0]])
  x0 = jnp.zeros(3)

  def prob_fn(rhs: jax.Array) -> px.CompositeProblem:
    return px.CompositeProblem(
        smooth=lambda x: 0.5 * jnp.sum(x**2) - jnp.dot(jnp.ones(3), x),
        proximal=px.prox.NonNegative(),
        constraints=px.constraints.equality(a_mat, rhs=rhs),
    )

  sol_vmap = px.batch_solve(
      prob_fn,
      x0,
      rhs_batch,
      method="ialm",
      opt_tol=1e-5,
      feas_tol=1e-5,
      backend="vmap",
      precondition=True,
  )
  sol_threads = px.batch_solve(
      prob_fn,
      x0,
      rhs_batch,
      method="ialm",
      opt_tol=1e-5,
      feas_tol=1e-5,
      backend="threads",
      precondition=True,
  )
  assert jnp.allclose(sol_vmap.x, sol_threads.x, atol=1e-4)
  assert jnp.all(sol_vmap.status == int(px.SolveStatus.CONVERGED))
