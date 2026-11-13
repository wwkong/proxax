"""Batched `jax.vmap` + `px.implicit_solve` throughput benchmarks."""

from __future__ import annotations

import json
import time

import jax
import jax.numpy as jnp

from proxax import px


def _capped_simplex_loss(
    scores: jax.Array, target: jax.Array
) -> jax.Array:
  def _prob(s: jax.Array) -> px.CompositeProblem:
    def _quad(x: jax.Array) -> jax.Array:
      return 0.5 * jnp.sum((x - s) ** 2)

    return px.CompositeProblem(
        smooth=px.oracle.as_smooth_term(_quad),
        proximal=px.prox.CappedSimplex(scale=3.0, upper=1.0),
    )

  x_hat = px.implicit_solve(
      _prob,
      jnp.full_like(scores, 3.0 / scores.shape[0]),
      scores,
      method="panoc",
      opt_tol=1e-4,
      linear_tol=1e-4,
  )
  return 0.5 * jnp.sum((x_hat - target) ** 2)


def _simplex_loss(scores: jax.Array, target: jax.Array) -> jax.Array:
  def _prob(s: jax.Array) -> px.CompositeProblem:
    def _quad(x: jax.Array) -> jax.Array:
      return 0.5 * jnp.sum((x - s) ** 2)

    return px.CompositeProblem(
        smooth=px.oracle.as_smooth_term(_quad),
        proximal=px.prox.Simplex(scale=1.0),
    )

  x_hat = px.implicit_solve(
      _prob,
      jnp.full_like(scores, 1.0 / scores.shape[0]),
      scores,
      method="panoc",
      opt_tol=1e-4,
      linear_tol=1e-4,
  )
  return 0.5 * jnp.sum((x_hat - target) ** 2)


def _bench_layer(name: str, loss_fn: object, dim: int = 16) -> list[dict]:
  batch_sizes = [1, 8, 32, 128, 256]
  vg_single = jax.value_and_grad(loss_fn)  # type: ignore[arg-type]

  def _batch_vg(scores_b: jax.Array, targets_b: jax.Array) -> tuple[
      jax.Array, jax.Array
  ]:
    losses, grads = jax.vmap(vg_single)(scores_b, targets_b)
    return jnp.mean(losses), grads

  batch_vg_jit = jax.jit(_batch_vg)
  rows = []
  key = jax.random.PRNGKey(0)
  for b in batch_sizes:
    k1, k2 = jax.random.split(jax.random.fold_in(key, b))
    scores_b = jax.random.normal(k1, (b, dim))
    targets_b = jax.random.uniform(k2, (b, dim))
    # Warmup JIT compilation
    loss, grad = batch_vg_jit(scores_b, targets_b)
    jax.block_until_ready((loss, grad))

    repeats = 10
    t0 = time.perf_counter()
    for _ in range(repeats):
      loss, grad = batch_vg_jit(scores_b, targets_b)
      jax.block_until_ready((loss, grad))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / repeats
    throughput = b / (elapsed_ms / 1000.0)
    rows.append({
        "layer": name,
        "batch_size": b,
        "ms_per_batch": round(elapsed_ms, 3),
        "samples_per_sec": round(throughput, 1),
    })
  return rows


def main() -> None:
  """Runs batched `vmap` implicit differentiation benchmarks."""
  rows = _bench_layer("CappedSimplex (PANOC)", _capped_simplex_loss)
  rows += _bench_layer("Simplex (PANOC)", _simplex_loss)
  print(json.dumps({"rows": rows}))


if __name__ == "__main__":
  main()
