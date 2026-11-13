"""End-to-end training with a differentiable ``px.layers.SimplexLayer``.

Demonstrates differentiating through a batch of simplex projections
(sparsemax) inside a JIT-compiled Optax training loop.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import optax

from proxax import px


def main() -> None:
    """Trains a linear-plus-SimplexLayer model to match sparse targets."""
    key = jax.random.PRNGKey(0)
    k_x, k_w, k_true = jax.random.split(key, 3)

    batch_size, d_in, d_out = 32, 6, 4
    features = jax.random.normal(k_x, (batch_size, d_in))

    # Ground-truth sparse probability vectors via SimplexLayer.
    w_true = jax.random.normal(k_true, (d_in, d_out))
    layer = px.layers.SimplexLayer(scale=1.0, opt_tol=1e-5, linear_tol=1e-5)
    targets = layer.batched(features @ w_true)

    params = {"W": 0.1 * jax.random.normal(k_w, (d_in, d_out))}
    optimizer = optax.adam(learning_rate=5e-2)
    opt_state = optimizer.init(params)

    def loss_fn(p: dict[str, jax.Array]) -> jax.Array:
        preds = layer.batched(features @ p["W"])
        return jnp.mean(jnp.sum((preds - targets) ** 2, axis=-1))

    @jax.jit
    def step(
        p: dict[str, jax.Array], state: optax.OptState
    ) -> tuple[dict[str, jax.Array], optax.OptState, jax.Array]:
        loss, grads = jax.value_and_grad(loss_fn)(p)
        updates, new_state = optimizer.update(grads, state, p)
        return optax.apply_updates(p, updates), new_state, loss

    for it in range(1, 61):
        params, opt_state, loss = step(params, opt_state)
        if it in (1, 20, 40, 60):
            print(f"step {it:>2d} | mse={float(loss):.4e}")


if __name__ == "__main__":
    main()
