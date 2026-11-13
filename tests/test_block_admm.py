"""Tests for BlockProblem and certified proximal ADMM."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from proxax import px


def test_admm_two_block_lasso_splitting() -> None:
  key = jax.random.key(0)
  a_mat = jax.random.normal(key, (6, 4))
  b_vec = jnp.array([1.0, -0.5, 0.2, 0.8, -1.2, 0.3])
  lam = 0.25

  block1 = px.BlockTerm(
      smooth=lambda x: 0.5 * jnp.sum((a_mat @ x - b_vec) ** 2),
      linear_op=px.constraints.identity,
  )
  block2 = px.BlockTerm(
      proximal=px.prox.L1(weight=lam),
      linear_op=-jnp.eye(4),
  )
  blk_prob = px.BlockProblem(blocks=(block1, block2), rhs=0.0)
  x0 = (jnp.zeros(4), jnp.zeros(4))
  sol = px.solve(blk_prob, x0, method="admm", opt_tol=1e-4, feas_tol=1e-4)
  assert sol.status == px.SolveStatus.CONVERGED
  assert sol.stats.opt_residual <= 1e-4
  assert sol.stats.feas_residual <= 1e-4
  assert jnp.allclose(sol.x[0], sol.x[1], atol=2e-4)

  # Compare against direct single-block APD solution
  direct = px.CompositeProblem(
      smooth=lambda x: 0.5 * jnp.sum((a_mat @ x - b_vec) ** 2),
      proximal=px.prox.L1(weight=lam),
  )
  ref = px.solve(direct, jnp.zeros(4), method="apd", opt_tol=1e-5)
  assert jnp.allclose(sol.x[0], ref.x, atol=1e-3)


def test_admm_three_block_and_eager_callback() -> None:
  rhs = jnp.array([1.0, -1.0, 0.5])
  b1 = px.BlockTerm(
      smooth=lambda x: 0.5 * jnp.sum((x - 1.0) ** 2),
      linear_op=px.constraints.identity,
  )
  b2 = px.BlockTerm(
      smooth=lambda x: 0.5 * jnp.sum((x + 0.5) ** 2),
      linear_op=px.constraints.identity,
  )
  b3 = px.BlockTerm(
      proximal=px.prox.Box(lower=-1.0, upper=1.0),
      linear_op=-jnp.eye(3),
  )
  prob = px.BlockProblem(blocks=(b1, b2, b3), rhs=rhs)
  x0 = (jnp.zeros(3), jnp.zeros(3), jnp.zeros(3))

  seen_iters: list[int] = []

  def _cb(it: int, stat: jax.Array, feas: jax.Array) -> None:
    del stat, feas
    seen_iters.append(it)

  solver = px.solvers.ADMM(opt_tol=1e-4, feas_tol=1e-4, rho=2.0)
  sol = px.solve(prob, x0, solver=solver, jit=False, callback=_cb)
  assert sol.status == px.SolveStatus.CONVERGED
  assert len(seen_iters) > 0
  assert jnp.allclose(sol.x[0] + sol.x[1] - sol.x[2], rhs, atol=1e-4)


def test_admm_general_linear_operators_and_vmap() -> None:
  a1 = jnp.array([[1.0, 2.0], [-1.0, 1.0]])
  a2 = jnp.array([[0.5, -1.0], [1.0, 0.5]])
  rhs = jnp.array([1.0, 2.0])

  b1 = px.BlockTerm(
      smooth=lambda x: 0.5 * jnp.sum(x**2),
      linear_op=a1,
  )
  b2 = px.BlockTerm(
      proximal=px.prox.NonNegative(),
      linear_op=a2,
  )
  prob = px.BlockProblem(blocks=(b1, b2), rhs=rhs)
  sol = px.solve(
      prob,
      (jnp.zeros(2), jnp.zeros(2)),
      solver=px.solvers.ADMM(opt_tol=1e-4, feas_tol=1e-4, rho=1.5),
  )
  assert sol.status == px.SolveStatus.CONVERGED
  assert jnp.all(sol.x[1] >= -1e-6)
  assert jnp.allclose(a1 @ sol.x[0] + a2 @ sol.x[1], rhs, atol=1e-4)
