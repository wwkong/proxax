# Optax Transforms & `jax.vmap`

Every algorithm in `proxax` is built from pure functional PyTree state machines
registered with `jax.tree_util.register_dataclass`. You can:

1. Drive `proxax` **one inner step at a time** inside a standard Optax
   `tx.init` / `tx.update` / `optax.apply_updates` loop via
   `proxax.optax_compat`.
2. Solve **batches of nonconvex optimization instances in parallel** on CPU,
   GPU, or TPU with `jax.jit(jax.vmap(...))`.

All code on this page can be executed directly via `examples/optax_and_vmap.py`:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python examples/optax_and_vmap.py
```

---

## 1. Step-by-Step Trajectory via `optax_compat.apd`

Unlike black-box solvers that only return the final iterate,
`proxax.optax_compat.apd` advances **one inner `PF.ACG` step** per `tx.update`
call while maintaining exact stationarity certificates `state.opt_residual`
($\|\hat{v}\|_2$) and adaptive weak-convexity estimates `state.lower_curvature`
($\hat{m}_k$) inside a `jax.jit`-compiled step function:

$$\min_{x \in [-1, 1]^4} \; \frac{1}{2}\sum_{i=1}^4 q_i x_i^2 + \langle c, x \rangle, \qquad q = (6, -2.5, 4, -1.5)$$

```python
import jax
import jax.numpy as jnp
import optax
from proxax import optax_compat
from proxax import prox

q_diag = jnp.array([6.0, -2.5, 4.0, -1.5])
linear_c = jnp.array([-1.2, 0.5, -0.8, 0.3])

def loss_fn(x: jax.Array) -> jax.Array:
  return 0.5 * jnp.sum(q_diag * (x**2)) + jnp.dot(linear_c, x)

tx = optax_compat.apd(
    prox_op=prox.Box(lower=-1.0, upper=1.0),
    opt_tol=1e-5,
    max_inner_iters=50,
)
params = jnp.array([0.8, 0.8, -0.5, -0.5])
state = tx.init(params)

@jax.jit
def step_fn(
    p: jax.Array,
    st: optax_compat.APDTransformState,
) -> tuple[jax.Array, optax_compat.APDTransformState]:
  val, grads = jax.value_and_grad(loss_fn)(p)
  updates, next_st = tx.update(grads, st, p, value=val, value_fn=loss_fn)
  return optax.apply_updates(p, updates), next_st

for step_idx in range(1, 80):
  params, state = step_fn(params, state)
  if bool(state.converged):
    break
```

### Verified Step-by-Step Trajectory

| Inner Step | Outer Stage | Objective $f(x_t)$ | Certified $\|\hat{v}\|_2$ | Adaptive $\hat{m}_k$ |
| ---: | ---: | ---: | ---: | ---: |
| `1` | `0` | `-1.637751` | `1.64e+00` | `1.000` |
| `3` | `1` | `-1.968868` | `8.39e-01` | `0.500` |
| `5` | `2` | `-1.998316` | `1.18e-01` | `0.250` |
| `10` | `3` | `-2.000000` | `1.95e-03` | `0.125` |
| `52` | `12` | `-2.000000` | `8.25e-06` | `0.250` |

---

## 2. Constrained Optax Loop via `optax_compat.ialm`

For set- or cone-constrained PyTree models $\mathcal{A}(w) \in \mathcal{C}$,
`proxax.optax_compat.ialm` (along with `qp_apd` and `qp_aipp`) tracks both the
primal iterate `params` and the dual multiplier `state.dual` inside an Optax
loop:

$$\min_{w \in [0, 1]^3} \; \frac{1}{2}\big(3 w_1^2 - w_2^2 + 2 w_3^2\big) - w_1 \quad \text{s.t.} \quad \sum_{i=1}^3 w_i = 1$$

```python
import jax
import jax.numpy as jnp
import optax
from proxax import constraints
from proxax import optax_compat
from proxax import prox

def smooth_fn(p: dict[str, jax.Array]) -> jax.Array:
  w = p["w"]
  return 0.5 * jnp.sum(jnp.array([3.0, -1.0, 2.0]) * (w**2)) - w[0]

tx = optax_compat.ialm(
    constraints=constraints.equality(
        lambda p: jnp.sum(p["w"]), rhs=jnp.asarray(1.0)
    ),
    prox_op=prox.Box(lower=0.0, upper=1.0),
    opt_tol=1e-4,
    feas_tol=1e-4,
)
params = {"w": jnp.array([0.2, 0.3, 0.1])}
state = tx.init(params)

@jax.jit
def step_fn(
    p: dict[str, jax.Array],
    st: optax_compat.IALMTransformState,
) -> tuple[dict[str, jax.Array], optax_compat.IALMTransformState]:
  val, grads = jax.value_and_grad(smooth_fn)(p)
  updates, next_st = tx.update(grads, st, p, value=val, value_fn=smooth_fn)
  return optax.apply_updates(p, updates), next_st

for _ in range(300):
  params, state = step_fn(params, state)
  if bool(state.converged):
    break
# converged=True, w*=[6.61e-05, 1.0, 0.0], stat=2.23e-05, feas=6.60e-05
```

You can also compose any `ProxOperator` directly inside an `optax.chain` for
standard proximal stochastic or deterministic gradient training via
`optax_compat.proximal`:

```python
tx_prox_sgd = optax.chain(
    optax.scale(-1e-2),
    optax_compat.proximal(prox.L1(weight=0.05), step_size=1e-2),
)
```

---

## 3. Batched Nonconvex Solves with `jax.jit(jax.vmap(...))`

Because `proxax` solvers compile their inner and outer loops with
`jax.lax.while_loop` (`jit=True`), `solver.solve` can be vectorized directly
over problem data `(Q_b, c_b)` using `jax.vmap`:

$$\min_{x \in \Delta^{10}} \; \frac{1}{2} x^\top Q_b x + \langle c_b, x \rangle, \qquad b = 1, \dots, 16$$

```python
import jax
import jax.numpy as jnp
import proxax
from proxax import prox
from proxax import solvers

batch_size, dim_n = 16, 10
k_q, k_c = jax.random.split(jax.random.key(123))
raw = jax.random.normal(k_q, (batch_size, dim_n, dim_n)) / jnp.sqrt(dim_n)
q_batch = 0.5 * (raw + jnp.swapaxes(raw, -1, -2)) - 0.8 * jnp.eye(dim_n)
c_batch = jax.random.normal(k_c, (batch_size, dim_n))
x0 = jnp.ones(dim_n) / dim_n
solver = solvers.APD(opt_tol=1e-4)

def solve_single(q_mat: jax.Array, c_vec: jax.Array) -> solvers.Solution:
  prob = proxax.CompositeProblem(
      smooth=lambda x: 0.5 * jnp.dot(x, q_mat @ x) + jnp.dot(c_vec, x),
      proximal=prox.Simplex(scale=1.0),
  )
  return solver.solve(prob, x0, jit=True)

batched_sol = jax.jit(jax.vmap(solve_single))(q_batch, c_batch)
# Solved 16/16 instances in parallel: max ||v_hat||_2 = 0.00e+00, mean inner iters = 7.7
```
