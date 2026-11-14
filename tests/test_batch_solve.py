"""Tests for `px.batch_solve` and `px.batch_implicit_solve`."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsparse

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


def test_batch_solve_batched_bcoo() -> None:
  """`batch_solve` supports batched `BCOO` matrices across vmap and threads."""
  x0 = jnp.zeros(4)
  a_batch = jsparse.BCOO.fromdense(
      jnp.stack([jnp.eye(4), 2.0 * jnp.eye(4), 3.0 * jnp.eye(4)]), n_batch=1
  )
  solver = IALM(opt_tol=1e-6, feas_tol=1e-6)

  def prob_fn(a: jsparse.BCOO) -> px.CompositeProblem:
    return px.CompositeProblem(
        smooth=lambda x: 0.5 * jnp.sum((x - 1.0) ** 2),
        constraints=px.constraints.equality(a, rhs=jnp.ones(4)),
    )

  sol_vmap = px.batch_solve(prob_fn, x0, a_batch, solver=solver, backend="vmap")
  sol_threads = px.batch_solve(
      prob_fn, x0, a_batch, solver=solver, backend="threads", max_workers=2
  )
  sol_chunked = px.batch_solve(
      prob_fn,
      x0,
      a_batch,
      solver=solver,
      backend="threads",
      max_workers=2,
      chunk_size=2,
  )
  assert jnp.allclose(sol_vmap.x, sol_threads.x, atol=1e-6)
  assert jnp.allclose(sol_vmap.x, sol_chunked.x, atol=1e-6)
  expected = jnp.stack([jnp.ones(4), 0.5 * jnp.ones(4), jnp.ones(4) / 3.0])
  assert jnp.allclose(sol_vmap.x, expected, atol=1e-4)


def test_batch_implicit_solve_batched_bcoo() -> None:
  """`batch_implicit_solve` supports batched `BCOO` across vmap and threads."""
  q_bcoo = jsparse.BCOO.fromdense(jnp.diag(jnp.array([2.0, 3.0, 4.0, 5.0])))
  a_dense = jnp.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
  a_batch = jsparse.BCOO.fromdense(
      jnp.stack([a_dense, 2.0 * a_dense, 0.5 * a_dense]), n_batch=1
  )
  c = jnp.array([-1.0, -2.0, -1.0, -2.0])
  b = jnp.array([1.0, 1.0])
  x0 = jnp.zeros(4)

  def prob_fn(a_mat: jsparse.BCOO) -> px.CompositeProblem:
    return px.CompositeProblem(
        smooth=lambda x: 0.5 * jnp.dot(x, q_bcoo @ x) + jnp.dot(c, x),
        constraints=px.constraints.equality(a_mat, rhs=b),
    )

  def loss(backend: str) -> typing.Any:
    def _fn(ab: jsparse.BCOO) -> jax.Array:
      xs = px.batch_implicit_solve(
          prob_fn,
          x0,
          ab,
          method="ialm",
          opt_tol=1e-6,
          feas_tol=1e-6,
          linear_tol=1e-6,
          backend=backend,
      )
      return 0.5 * jnp.sum(xs**2)

    return _fn

  v_vmap, g_vmap = jax.value_and_grad(loss("vmap"), allow_int=True)(a_batch)
  v_threads, g_threads = jax.value_and_grad(loss("threads"), allow_int=True)(
      a_batch
  )
  assert jnp.allclose(v_vmap, v_threads, atol=1e-3)
  assert jnp.allclose(g_vmap.data, g_threads.data, atol=1e-3)
