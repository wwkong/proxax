# Batching & Parallelism

Proxax supports two complementary ways to solve batches of parameterized
optimization problems in parallel:

1. **Vectorized XLA (`jax.vmap` / `backend="vmap"`):** Compiles a single SIMD/GPU
   graph across the batch. Best on **GPU/TPU**, inside outer `jax.lax.scan`
   loops, or when batch elements converge in uniform iteration counts.
2. **CPU Thread-Pool (`backend="threads"`):** Compiles one per-instance (or
   per-chunk) XLA executable and dispatches work across a persistent
   `ThreadPoolExecutor`, letting easy instances exit early on CPU without
   waiting for the slowest element in the batch.

---

## 1. Manual `jax.jit(jax.vmap(...))`

Because Proxax compiles solver loops with `jax.lax.while_loop` (`jit=True`),
you can wrap `solver.solve` directly in `jax.vmap` to solve dozens or thousands
of parameterized optimization instances simultaneously on an accelerator:

$$\min_{x \in \Delta^{10}} \; \frac{1}{2} x^\top Q_b x + \langle c_b, x \rangle, \qquad b = 1, \dots, 16$$

```python
import jax
import jax.numpy as jnp
from proxax import px

batch_size, dim_n = 16, 10
k_q, k_c = jax.random.split(jax.random.key(123))
raw = jax.random.normal(k_q, (batch_size, dim_n, dim_n)) / jnp.sqrt(dim_n)
q_batch = 0.5 * (raw + jnp.swapaxes(raw, -1, -2)) - 0.8 * jnp.eye(dim_n)
c_batch = jax.random.normal(k_c, (batch_size, dim_n))
x0 = jnp.ones(dim_n) / dim_n
solver = px.solvers.APD(opt_tol=1e-4)

def solve_single(q_mat: jax.Array, c_vec: jax.Array) -> px.Solution:
  def quad_fn(x: jax.Array) -> jax.Array:
    return 0.5 * jnp.dot(x, q_mat @ x) + jnp.dot(c_vec, x)

  prob = px.CompositeProblem(
      smooth=quad_fn,
      proximal=px.prox.Simplex(scale=1.0),
  )
  return solver.solve(prob, x0, jit=True)

batched_sol = jax.jit(jax.vmap(solve_single))(q_batch, c_batch)
print("Statuses:", batched_sol.status)          # shape (16,), all 0 (CONVERGED)
print("Solutions shape:", batched_sol.x.shape)  # shape (16, 10)
```

---

## 2. High-Level Batching: `px.batch_solve` & `px.batch_implicit_solve`

For convenience—and especially on **CPU** where `jax.vmap` executes all $B$
instances in lockstep up to $\max_i k_i$ iterations—Proxax provides
`px.batch_solve` (forward-only) and `px.batch_implicit_solve` (differentiable):

```python
def make_prob(c_vec: jax.Array) -> px.CompositeProblem:
  return px.CompositeProblem(
      smooth=lambda x: 0.5 * jnp.dot(x, Q0 @ x) + jnp.dot(c_vec, x),
      proximal=px.prox.Simplex(scale=1.0),
  )

# Forward-only batched solve (backend="auto" selects "threads" on CPU for B > 1, "vmap" on GPU/TPU)
sols = px.batch_solve(make_prob, x0, c_batch, solver=solver, chunk_size=1)

# Differentiable batched solve (forward via backend, implicit VJP via vmap)
def loss_fn(theta: jax.Array) -> jax.Array:
  xs = px.batch_implicit_solve(make_prob, x0, c_batch + theta[None, :], solver=solver)
  return 0.5 * jnp.mean(jnp.sum((xs - x_target) ** 2, axis=-1))
```

### Choosing `backend` and `chunk_size`

* **`backend="vmap"`:** Best on **GPU/TPU**, inside outer `jax.lax.scan`
  training loops, or on CPU when iteration counts are tight and uniform across
  the batch.
* **`backend="threads"` (default on CPU when `B > 1`):** Compiles one
  per-instance (or per-chunk) XLA executable and runs elements concurrently on
  a persistent `ThreadPoolExecutor`, letting easy instances exit early instead
  of waiting for the hardest element in the batch.
* **`chunk_size` (`px.batch_solve`):** `chunk_size=1` minimizes straggler
  lockstep on heterogeneous active-set problems; `chunk_size=4` (or `8`)
  amortizes host-dispatch overhead on uniform CPU batches.
* **Python training loops with `backend="threads"`:** Calling
  `jax.value_and_grad(loss_fn)` eagerly (without an outer `jax.jit`) bypasses
  `jax.pure_callback` and works out of the box. If you wrap
  `jax.value_and_grad(loss_fn)` in an outer `jax.jit` inside a Python `for`
  loop, call `grad.block_until_ready()` before dispatching the next optimizer
  update so JAX's async CPU execution does not overlap downstream kernels with
  `pure_callback` worker threads.

See {doc}`../appendices/implementation_optimizations` for caching and
implementation details.
