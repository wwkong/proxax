# Optax Integration & `jax.vmap`

Every operator and solver in `proxax` is implemented using pure functional JAX
PyTree state machines. You can:

1. Plug any `ProxOperator` into an `optax.chain` in 3 lines via
   `px.optax_compat.proximal`.
2. Step through certified nonconvex solvers (`apd`, `aipp`, `ialm`, `qp_apd`,
   `qp_aipp`) **one inner iteration at a time** inside a standard Optax loop.
3. Solve **batches of independent optimization problems in parallel** on GPU or
   TPU using `jax.jit(jax.vmap(...))`.

All examples on this page can be run via `examples/optax_and_vmap.py`:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python examples/optax_and_vmap.py
```

---

## 1. Proximal Gradient / SGD in `optax.chain` (`px.optax_compat.proximal`)

When training a model with Optax and you want every parameter update to apply a
proximal operator (such as soft-thresholding or simplex/box projection), append
`px.optax_compat.proximal(prox_op, step_size)` to `optax.chain` and pass
`params` to `tx.update`:

```python
import jax
import jax.numpy as jnp
import optax
from proxax import px

lr = 1e-2
tx = optax.chain(
    optax.scale(-lr),
    px.optax_compat.proximal(px.prox.L1(weight=0.05), step_size=lr),
)

params = jnp.array([0.5, -0.02, 1.2])
state = tx.init(params)

grads = jnp.array([0.1, 0.05, -0.3])
updates, state = tx.update(grads, state, params)
params = optax.apply_updates(params, updates)
```

`step_size` can be either a float scalar or an `optax.Schedule`.

---

## 2. Certified Unconstrained Optax Loops (`px.optax_compat.apd`)

If you want to inspect iterates, log custom metrics, or interleave solver steps
with external JAX simulation code, `px.optax_compat.apd` (and
`px.optax_compat.aipp`) advances **one inner step per `tx.update` call** while
tracking the exact stationarity residual `state.opt_residual` and adaptive
curvature estimate `state.lower_curvature`:

$$\min_{x \in [-1, 1]^4} \; \frac{1}{2}\sum_{i=1}^4 q_i x_i^2 + \langle c, x \rangle, \qquad q = (6, -2.5, 4, -1.5)$$

```python
import jax
import jax.numpy as jnp
import optax
from proxax import px

q_diag = jnp.array([6.0, -2.5, 4.0, -1.5])
linear_c = jnp.array([-1.2, 0.5, -0.8, 0.3])

def loss_fn(x: jax.Array) -> jax.Array:
  return 0.5 * jnp.sum(q_diag * (x**2)) + jnp.dot(linear_c, x)

tx = px.optax_compat.apd(
    prox_op=px.prox.Box(lower=-1.0, upper=1.0),
    opt_tol=1e-5,
    max_inner_iters=50,
)
params = jnp.array([0.8, 0.8, -0.5, -0.5])
state = tx.init(params)

@jax.jit
def step_fn(
    p: jax.Array,
    st: px.optax_compat.APDTransformState,
) -> tuple[jax.Array, px.optax_compat.APDTransformState]:
  val, grads = jax.value_and_grad(loss_fn)(p)
  updates, next_st = tx.update(grads, st, p, value=val, value_fn=loss_fn)
  return optax.apply_updates(p, updates), next_st

for _ in range(80):
  params, state = step_fn(params, state)
  if bool(state.converged):
    break
```

| Inner Step | Outer Stage | Objective $f(x_t)$ | Certified $\|\hat{v}\|_2$ | Adaptive $\hat{m}_k$ |
| ---: | ---: | ---: | ---: | ---: |
| `1` | `0` | `-1.637751` | `1.64e+00` | `1.000` |
| `3` | `1` | `-1.968868` | `8.39e-01` | `0.500` |
| `5` | `2` | `-1.998316` | `1.18e-01` | `0.250` |
| `10` | `3` | `-2.000000` | `1.95e-03` | `0.125` |
| `52` | `12` | `-2.000000` | `8.25e-06` | `0.250` |

---

## 3. Constrained Optax Loops (`px.optax_compat.ialm`, `qp_apd`, `qp_aipp`)

For problems with linear or cone constraints $\mathcal{A}(w) \in \mathcal{C}$,
`px.optax_compat.ialm` (along with `qp_apd` and `qp_aipp`) updates both `params`
and the Lagrange multipliers `state.dual` inside an Optax step function:

$$\min_{w \in [0, 1]^3} \; \frac{1}{2}\big(3 w_1^2 - w_2^2 + 2 w_3^2\big) - w_1 \quad \text{s.t.} \quad \sum_{i=1}^3 w_i = 1$$

```python
import jax
import jax.numpy as jnp
import optax
from proxax import px

def smooth_fn(p: dict[str, jax.Array]) -> jax.Array:
  w = p["w"]
  return 0.5 * jnp.sum(jnp.array([3.0, -1.0, 2.0]) * (w**2)) - w[0]

tx = px.optax_compat.ialm(
    constraints=px.constraints.equality(
        lambda p: jnp.sum(p["w"]), rhs=jnp.asarray(1.0)
    ),
    prox_op=px.prox.Box(lower=0.0, upper=1.0),
    opt_tol=1e-4,
    feas_tol=1e-4,
)
params = {"w": jnp.array([0.2, 0.3, 0.1])}
state = tx.init(params)

@jax.jit
def step_fn(
    p: dict[str, jax.Array],
    st: px.optax_compat.IALMTransformState,
) -> tuple[dict[str, jax.Array], px.optax_compat.IALMTransformState]:
  val, grads = jax.value_and_grad(smooth_fn)(p)
  updates, next_st = tx.update(grads, st, p, value=val, value_fn=smooth_fn)
  return optax.apply_updates(p, updates), next_st

for _ in range(300):
  params, state = step_fn(params, state)
  if bool(state.converged):
    break
# converged=True, w*=[6.61e-05, 1.0, 0.0], stat=2.23e-05, feas=6.60e-05
```

---

## 4. Batched Solves with `jax.jit(jax.vmap(...))`

Because `proxax` compiles solver loops with `jax.lax.while_loop` (`jit=True`),
you can wrap `solver.solve` in `jax.vmap` to solve dozens or thousands of
parameterized optimization instances simultaneously on an accelerator:

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
  prob = px.CompositeProblem(
      smooth=lambda x: 0.5 * jnp.dot(x, q_mat @ x) + jnp.dot(c_vec, x),
      proximal=px.prox.Simplex(scale=1.0),
  )
  return solver.solve(prob, x0, jit=True)

batched_sol = jax.jit(jax.vmap(solve_single))(q_batch, c_batch)
print("Statuses:", batched_sol.status)          # shape (16,), all 0 (CONVERGED)
print("Solutions shape:", batched_sol.x.shape)  # shape (16, 10)
```
