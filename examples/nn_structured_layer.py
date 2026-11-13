"""End-to-end MLP with a differentiable `CappedSimplex` top-k layer."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp
import optax

from proxax import px


def _capped_simplex_layer(
    scores: jax.Array, *, k: float = 2.0
) -> jax.Array:
  """Projects scores onto `{0 <= x <= 1, sum(x) = k}` with `implicit_solve`."""

  def _prob(s: jax.Array) -> px.CompositeProblem:
    def _quad(x: jax.Array) -> jax.Array:
      return 0.5 * jnp.sum((x - s) ** 2)

    return px.CompositeProblem(
        smooth=px.oracle.as_smooth_term(_quad),
        proximal=px.prox.CappedSimplex(scale=k, upper=1.0),
    )

  return typing.cast(
      jax.Array,
      px.implicit_solve(
          _prob,
          jnp.full_like(scores, k / scores.shape[0]),
          scores,
          method="panoc",
          opt_tol=1e-4,
          linear_tol=1e-4,
      ),
  )


def _forward_single(
    params: dict[str, jax.Array], x: jax.Array
) -> jax.Array:
  h = jax.nn.tanh(x @ params["w1"] + params["b1"])
  logits = h @ params["w2"] + params["b2"]
  return _capped_simplex_layer(logits, k=2.0)


def main() -> None:
  """Trains a 2-layer MLP + `CappedSimplex` layer on synthetic top-2 data."""
  key = jax.random.PRNGKey(0)
  k1, k2, k3, k4 = jax.random.split(key, 4)
  d_in, d_hid, d_out, n_samples = 4, 8, 5, 32

  X = jax.random.normal(k1, (n_samples, d_in))
  # Target: top-2 indices of a hidden linear rule
  true_w = jax.random.normal(k2, (d_in, d_out))
  _, top2 = jax.lax.top_k(X @ true_w, 2)
  Y = jnp.zeros((n_samples, d_out)).at[
      jnp.arange(n_samples)[:, None], top2
  ].set(1.0)

  params = {
      "w1": 0.3 * jax.random.normal(k3, (d_in, d_hid)),
      "b1": jnp.zeros(d_hid),
      "w2": 0.3 * jax.random.normal(k4, (d_hid, d_out)),
      "b2": jnp.zeros(d_out),
  }

  def _loss_fn(p: dict[str, jax.Array]) -> jax.Array:
    preds = jax.vmap(lambda x: _forward_single(p, x))(X)
    return 0.5 * jnp.mean(jnp.sum((preds - Y) ** 2, axis=-1))

  opt = optax.adam(learning_rate=0.05)
  opt_state = opt.init(params)
  step_fn = jax.jit(jax.value_and_grad(_loss_fn))

  for step in range(1, 31):
    loss, grads = step_fn(params)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    if step in (1, 15, 30):
      print(f"Step {step:2d} | MSE={float(loss):.6f}")


if __name__ == "__main__":
  main()
