"""Batched forward and differentiable solves via ``px.batch_solve``.

Demonstrates ``px.batch_solve`` (``backend="vmap"`` vs. ``backend="threads"``
with ``chunk_size``) and eager ``jax.value_and_grad`` training using
``px.batch_implicit_solve(..., backend="threads")``.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

from proxax import px


def main() -> None:
  """Runs batched forward and differentiable simplex QP examples."""
  key = jax.random.key(0)
  k_q, k_c = jax.random.split(key)
  batch_size, dim = 16, 12

  raw = jax.random.normal(k_q, (dim, dim)) / jnp.sqrt(dim)
  q_mat = raw.T @ raw + 0.5 * jnp.eye(dim)
  c_batch = jax.random.normal(k_c, (batch_size, dim))
  x0 = jnp.full(dim, 1.0 / dim)
  solver = px.solvers.PANOC(opt_tol=1e-5)

  def make_prob(c_vec: jax.Array) -> px.CompositeProblem:
    return px.CompositeProblem(
        smooth=lambda x: 0.5 * jnp.dot(x, q_mat @ x) + jnp.dot(c_vec, x),
        proximal=px.prox.Simplex(scale=1.0),
    )

  sol_vmap = px.batch_solve(
      make_prob, x0, c_batch, solver=solver, backend="vmap"
  )
  sol_threads = px.batch_solve(
      make_prob,
      x0,
      c_batch,
      solver=solver,
      backend="threads",
      chunk_size=4,
  )
  max_diff = float(jnp.max(jnp.abs(sol_vmap.x - sol_threads.x)))
  print(f"batch_solve max |vmap - threads| = {max_diff:.2e}")

  x_target = jnp.full((batch_size, dim), 1.0 / dim)

  def loss_fn(theta: jax.Array) -> jax.Array:
    xs = px.batch_implicit_solve(
        make_prob,
        x0,
        c_batch + theta[None, :],
        solver=solver,
        opt_tol=1e-5,
        linear_tol=1e-5,
        backend="threads",
    )
    return 0.5 * jnp.mean(jnp.sum((xs - x_target) ** 2, axis=-1))

  theta = jnp.zeros(dim)
  val_and_grad = jax.value_and_grad(loss_fn)
  for step in range(1, 6):
    loss_val, grad = val_and_grad(theta)
    theta = theta - 0.2 * grad
    print(f"step {step} | loss={float(loss_val):.6f}")


if __name__ == "__main__":
  main()
