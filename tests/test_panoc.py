"""Tests for the composite quasi-Newton `PANOC` solver."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from proxax import px


def test_panoc_ill_conditioned_lasso_and_jit() -> None:
  """`PANOC` solves ill-conditioned Lasso under `jit=True` and `jit=False`."""
  diag = jnp.array([100.0, 20.0, 2.0, 0.2, 0.05])
  target = jnp.array([1.5, -1.0, 0.5, -0.2, 0.8])

  def _loss(x: jax.Array) -> jax.Array:
    return 0.5 * jnp.sum(diag * (x - target) ** 2)

  prob = px.CompositeProblem(
      smooth=px.oracle.as_smooth_term(_loss),
      proximal=px.prox.L1(weight=0.1),
  )
  sol_jit = px.solve(prob, jnp.zeros(5), method="panoc", opt_tol=1e-5)
  assert sol_jit.status_enum == px.SolveStatus.CONVERGED
  assert float(sol_jit.stats.opt_residual) <= 1e-5

  sol_eager = px.solve(
      prob, jnp.zeros(5), method="panoc", opt_tol=1e-5, jit=False
  )
  assert sol_eager.status_enum == px.SolveStatus.CONVERGED
  assert jnp.allclose(sol_jit.x, sol_eager.x, atol=1e-4)


def test_panoc_nonconvex_box_and_exotic_prox() -> None:
  """`PANOC` handles nonconvex objectives with `Box` and `Spectrahedron`."""
  q_mat = jnp.array([[4.0, -2.0], [-2.0, -1.5]])
  c_vec = jnp.array([-1.0, 0.5])

  def _nc_quad(x: jax.Array) -> jax.Array:
    return 0.5 * jnp.dot(x, q_mat @ x) + jnp.dot(c_vec, x)

  prob = px.CompositeProblem(
      smooth=px.oracle.as_smooth_term(_nc_quad),
      proximal=px.prox.Box(lower=-1.0, upper=1.0),
  )
  sol = px.solve(prob, jnp.zeros(2), method="panoc", opt_tol=1e-5)
  assert sol.status_enum == px.SolveStatus.CONVERGED
  assert jnp.all((sol.x >= -1.0 - 1e-6) & (sol.x <= 1.0 + 1e-6))

  target_mat = jnp.array([[2.0, 0.8], [0.8, -0.5]])

  def _spec_loss(x: jax.Array) -> jax.Array:
    return 0.5 * jnp.sum((x - target_mat) ** 2)

  spec_prob = px.CompositeProblem(
      smooth=px.oracle.as_smooth_term(_spec_loss),
      proximal=px.prox.Spectrahedron(scale=1.0),
  )
  sol_spec = px.solve(
      spec_prob, 0.5 * jnp.eye(2), method="panoc", opt_tol=1e-5
  )
  assert sol_spec.status_enum == px.SolveStatus.CONVERGED
  assert jnp.allclose(jnp.trace(sol_spec.x), 1.0, atol=1e-5)
  assert jnp.all(jnp.linalg.eigvalsh(sol_spec.x) >= -1e-6)


def test_panoc_vmap_batching() -> None:
  """`PANOC.solve` composes cleanly with `jax.jit(jax.vmap(...))`."""
  solver = px.solvers.PANOC(opt_tol=1e-4)
  targets = jnp.array([[1.0, -2.0, 0.5], [-1.5, 0.3, 2.0]])

  def _solve_one(tgt: jax.Array) -> px.Solution:
    def _loss(x: jax.Array) -> jax.Array:
      return 0.5 * jnp.sum((x - tgt) ** 2)

    prob = px.CompositeProblem(
        smooth=px.oracle.as_smooth_term(_loss),
        proximal=px.prox.L1(weight=0.1),
    )
    return solver.solve(prob, jnp.zeros(3), jit=True)

  batched = jax.jit(jax.vmap(_solve_one))(targets)
  assert batched.status.shape == (2,)
  assert jnp.all(batched.status == 0)
