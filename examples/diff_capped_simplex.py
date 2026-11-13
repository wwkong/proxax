"""Differentiable `CappedSimplex` top-k attention via `px.implicit_solve`."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp
import optax

from proxax import px


def capped_simplex_layer(
    scores: jax.Array, *, total: float = 2.0, upper: float = 1.0
) -> jax.Array:
  """Differentiable projection onto `{0 <= x <= upper, sum(x) = total}`."""

  def _proj_problem(s: jax.Array) -> px.CompositeProblem:
    def _quad(x: jax.Array) -> jax.Array:
      return 0.5 * jnp.sum((x - s) ** 2)

    return px.CompositeProblem(
        smooth=px.oracle.as_smooth_term(_quad),
        proximal=px.prox.CappedSimplex(scale=total, upper=upper),
    )

  return typing.cast(
      jax.Array,
      px.implicit_solve(
          _proj_problem,
          jnp.full_like(scores, total / scores.shape[0]),
          scores,
          method="panoc",
          opt_tol=1e-5,
          linear_tol=1e-5,
      ),
  )


def main() -> None:
  """Trains a linear encoder + `CappedSimplex` layer with `optax.adam`."""
  key = jax.random.PRNGKey(42)
  k1, k2 = jax.random.split(key)
  n_items, feat_dim = 6, 4
  features = jax.random.normal(k1, (n_items, feat_dim))
  # Target: select items 0 and 3 with weight 1.0 each (total=2.0, upper=1.0).
  target_weights = jnp.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])

  def loss_fn(params: tuple[jax.Array, jax.Array]) -> jax.Array:
    w, b = params
    scores = features @ w + b
    attn = capped_simplex_layer(scores, total=2.0, upper=1.0)
    return 0.5 * jnp.sum((attn - target_weights) ** 2)

  params = (0.1 * jax.random.normal(k2, (feat_dim,)), jnp.zeros(n_items))
  optimizer = optax.adam(learning_rate=0.2)
  opt_state = optimizer.init(params)
  step_fn = jax.jit(jax.value_and_grad(loss_fn))

  init_loss, _ = step_fn(params)
  print(f"Initial loss: {float(init_loss):.6f}")
  for step in range(1, 26):
    lv, grads = step_fn(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    if step in (1, 10, 25):
      print(f"Step {step:2d} | loss={float(lv):.6f}")

  final_scores = features @ params[0] + params[1]
  final_attn = capped_simplex_layer(final_scores, total=2.0, upper=1.0)
  print("Learned attention:", jnp.round(final_attn, 4))


if __name__ == "__main__":
  main()
