# Differentiable Optimization Layers

This example trains a linear model composed with a differentiable
`px.layers.SimplexLayer` (sparsemax) inside a `jax.jit`-compiled Optax
training loop. Gradients flow through the batched simplex projections via
`px.implicit_solve`.

## Runnable Script (`examples/layers_training.py`)

```python
import jax
import jax.numpy as jnp
import optax
from proxax import px

key = jax.random.PRNGKey(0)
k_x, k_w, k_true = jax.random.split(key, 3)

batch_size, d_in, d_out = 32, 6, 4
features = jax.random.normal(k_x, (batch_size, d_in))

w_true = jax.random.normal(k_true, (d_in, d_out))
layer = px.layers.SimplexLayer(scale=1.0, opt_tol=1e-5, linear_tol=1e-5)
targets = layer.batched(features @ w_true)

params = {"W": 0.1 * jax.random.normal(k_w, (d_in, d_out))}
optimizer = optax.adam(learning_rate=5e-2)
opt_state = optimizer.init(params)


def loss_fn(p):
  preds = layer.batched(features @ p["W"])
  return jnp.mean(jnp.sum((preds - targets) ** 2, axis=-1))


@jax.jit
def step(p, state):
  loss, grads = jax.value_and_grad(loss_fn)(p)
  updates, new_state = optimizer.update(grads, state, p)
  return optax.apply_updates(p, updates), new_state, loss


for it in range(1, 61):
  params, opt_state, loss = step(params, opt_state)
  if it in (1, 20, 40, 60):
    print(f"step {it:>2d} | mse={float(loss):.4e}")
```

## Sample Output

```text
step  1 | mse=9.5961e-01
step 20 | mse=6.5578e-02
step 40 | mse=3.9828e-02
step 60 | mse=3.4042e-02
```
