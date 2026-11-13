"""Bilevel hyperparameter learning via `px.implicit_solve` and Optax."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from proxax import px


def main() -> None:
  """Learns Elastic-Net hyperparameters on a sparse validation loss."""
  key = jax.random.PRNGKey(0)
  k1, k2, k3, k4 = jax.random.split(key, 4)
  n_train, n_val, dim = 40, 40, 12
  x_true = jnp.array(
      [2.0, -1.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  )

  a_train = jax.random.normal(k1, (n_train, dim)) / jnp.sqrt(n_train)
  y_train = a_train @ x_true + 0.15 * jax.random.normal(k2, (n_train,))
  a_val = jax.random.normal(k3, (n_val, dim)) / jnp.sqrt(n_val)
  y_val = a_val @ x_true + 0.15 * jax.random.normal(k4, (n_val,))

  def make_elastic_net(
      params: tuple[jax.Array, jax.Array],
  ) -> px.CompositeProblem:
    log_l1, log_l2 = params
    l2_weights = jnp.exp(log_l2)

    def _loss(x: jax.Array) -> jax.Array:
      return 0.5 * jnp.sum((a_train @ x - y_train) ** 2) + 0.5 * jnp.sum(
          l2_weights * (x**2)
      )

    return px.CompositeProblem(
        smooth=px.oracle.as_smooth_term(_loss),
        proximal=px.prox.L1(weight=jnp.exp(log_l1)),
    )

  def val_objective(params: tuple[jax.Array, jax.Array]) -> jax.Array:
    x_hat = px.implicit_solve(
        make_elastic_net,
        jnp.zeros(dim),
        params,
        method="apd",
        opt_tol=1e-5,
        linear_tol=1e-5,
    )
    return 0.5 * jnp.sum((a_val @ x_hat - y_val) ** 2)

  params = (jnp.log(0.05), jnp.full(dim, jnp.log(0.05)))
  optimizer = optax.adam(learning_rate=0.15)
  opt_state = optimizer.init(params)
  step_fn = jax.jit(jax.value_and_grad(val_objective))

  init_loss, _ = step_fn(params)
  print(f"Initial validation loss: {float(init_loss):.6f}")

  for step in range(1, 31):
    loss_val, grads = step_fn(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    if step in (1, 10, 20, 30):
      print(f"Step {step:2d} | val_loss={float(loss_val):.6f}")

  final_loss, _ = step_fn(params)
  print(f"Final validation loss:   {float(final_loss):.6f}")
  print(f"Learned L1 weight:       {float(jnp.exp(params[0])):.4f}")
  print("Learned L2 weights (first 6):", jnp.round(jnp.exp(params[1])[:6], 4))


if __name__ == "__main__":
  main()
