"""Tests for `px.batch_solve` and `px.batch_implicit_solve`."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from proxax import px
from proxax.solvers import IALM
from proxax.solvers import PANOC


def test_batch_solve_threads_matches_vmap() -> None:
  """`batch_solve(..., backend='threads')` matches `backend='vmap'`."""
  n = 10
  P = jnp.eye(n)
  q_batch = jax.random.normal(jax.random.PRNGKey(0), (6, n))
  x0 = jnp.full(n, 1.0 / n)

  def prob_fn(q: jax.Array) -> px.CompositeProblem:
    return px.CompositeProblem(
        smooth=lambda x: 0.5 * jnp.dot(x, P @ x) + jnp.dot(q, x),
        proximal=px.prox.Simplex(),
    )

  sol_vmap = px.batch_solve(
      prob_fn,
      x0,
      q_batch,
      solver=PANOC(opt_tol=1e-6),
      opt_tol=1e-6,
      backend="vmap",
  )
  sol_threads = px.batch_solve(
      prob_fn,
      x0,
      q_batch,
      solver=PANOC(opt_tol=1e-6),
      opt_tol=1e-6,
      backend="threads",
      max_workers=4,
  )
  sol_chunked = px.batch_solve(
      prob_fn,
      x0,
      q_batch,
      solver=PANOC(opt_tol=1e-6),
      opt_tol=1e-6,
      backend="threads",
      max_workers=2,
      chunk_size=4,
  )
  assert jnp.allclose(sol_vmap.x, sol_threads.x, atol=1e-6)
  assert jnp.allclose(sol_vmap.x, sol_chunked.x, atol=1e-6)


def test_batch_implicit_solve_threads_matches_vmap() -> None:
  """`batch_implicit_solve(..., backend='threads')` matches `vmap` grads."""
  n, m = 12, 5
  key = jax.random.PRNGKey(1)
  P = jnp.eye(n)
  A = jax.random.normal(key, (m, n)) / jnp.sqrt(n)
  b = A @ jnp.full(n, 0.2) + 0.5
  q_batch = jax.random.normal(jax.random.PRNGKey(2), (8, n))
  x0 = jnp.full(n, 0.2)
  solver = IALM(opt_tol=1e-6, feas_tol=1e-6)

  def prob_fn(q: jax.Array) -> px.CompositeProblem:
    return px.CompositeProblem(
        smooth=lambda x: 0.5 * jnp.dot(x, P @ x) + jnp.dot(q, x),
        proximal=px.prox.NonNegative(),
        constraints=px.constraints.less_equal(A, b),
    )

  def loss_vmap(qb: jax.Array) -> jax.Array:
    xs = px.batch_implicit_solve(
        prob_fn,
        x0,
        qb,
        solver=solver,
        opt_tol=1e-6,
        feas_tol=1e-6,
        linear_tol=1e-6,
        backend="vmap",
    )
    return 0.5 * jnp.sum(xs**2)

  def loss_threads(qb: jax.Array) -> jax.Array:
    xs = px.batch_implicit_solve(
        prob_fn,
        x0,
        qb,
        solver=solver,
        opt_tol=1e-6,
        feas_tol=1e-6,
        linear_tol=1e-6,
        backend="threads",
        max_workers=4,
    )
    return 0.5 * jnp.sum(xs**2)

  v1, g1 = jax.jit(jax.value_and_grad(loss_vmap))(q_batch)
  v2, g2 = jax.jit(jax.value_and_grad(loss_threads))(q_batch)
  assert jnp.allclose(v1, v2, atol=1e-5)
  assert jnp.allclose(g1, g2, atol=1e-4)

  # Eager repeated calls exercise _BATCH_IMPLICIT_CACHE reuse across steps.
  eager_vg = jax.value_and_grad(loss_threads)
  for _ in range(2):
    ve, ge = eager_vg(q_batch)
    assert jnp.allclose(v1, ve, atol=1e-5)
    assert jnp.allclose(g1, ge, atol=1e-4)
