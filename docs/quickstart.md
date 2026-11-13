# Installation & Quickstart

Proxax solves smooth, nonsmooth, and constrained optimization problems in JAX.
You write ordinary differentiable JAX functions for the smooth part of your
objective, pick built-in proximal operators and constraints for the nonsmooth or
constrained parts, and call `px.solve`.

---

## Installation

::::{tab-set}

:::{tab-item} pip (CPU)
```bash
pip install proxax
```
:::

:::{tab-item} uv
```bash
uv add proxax
```
:::

:::{tab-item} NVIDIA GPU (CUDA 12 / 13)
```bash
# CUDA 13
pip install "proxax[cuda13]"

# CUDA 12
pip install "proxax[cuda12]"
```
:::

::::

### Importing Proxax

Everything in Proxax is exposed under a single curated namespace `px` —
either `from proxax import px` (or `import proxax as px`):

- **Top-level entry points:** `px.CompositeProblem`, `px.SpectralProblem`,
  `px.solve`, `px.Solution`, `px.SolveStatus`
- **Submodules:** `px.prox`, `px.constraints`, `px.solvers`, `px.oracle`,
  `px.optax_compat`, `px.certificates`, `px.tree`, `px.typing`

---

## 1. Sparse Optimization (`smooth` + `px.prox.L1`)

Suppose you want to minimize a smooth loss plus an $\ell_1$ sparsity penalty:

$$\min_{x \in \mathbb{R}^n} \; \frac{1}{2}\|A x - b\|_2^2 + \lambda \|x\|_1$$

Pass your JAX loss function as `smooth` and `px.prox.L1(weight=...)` as
`proximal`:

```python
import jax.numpy as jnp
from proxax import px

a_mat = jnp.array([[1.0, 0.5, -0.2], [0.0, 1.2, 0.8]])
b_vec = jnp.array([1.5, -0.6])

def loss_fn(x):
  return 0.5 * jnp.sum((a_mat @ x - b_vec) ** 2)


problem = px.CompositeProblem(
    smooth=loss_fn,
    proximal=px.prox.L1(weight=0.1),
)

sol = px.solve(problem=problem, x0=jnp.zeros(3), method="apd", opt_tol=1e-6)
print(sol)
print("x =", sol.x)
```

**Output:**

```text
Solution(
  status=CONVERGED,
  obj=2.000000e-01,
  opt_res=9.479e-07,
  feas_res=0.000e+00,
  iters=12 (inner=82),
  evals=(smooth=200, grad=174, prox=88),
  runtime_ms=0.29,
  x=float32[3],
  dual=None,
)
x = [ 1.2749995 -0.        -0.6249994]
```

You do not need to supply gradients or step sizes: Proxax differentiates
`smooth` automatically with `jax.value_and_grad` and estimates local smoothness
and curvature on the fly.

---

## 2. Optimizing Over Simple Sets (`Box`, `Simplex`, `NonNegative`)

When your variable must stay inside a simple set whose Euclidean projection is
fast — such as a box $[l, u]$, the non-negative orthant $x \ge 0$, or the
probability simplex $\{x \ge 0 : \sum_i x_i = 1\}$ — encode that set directly
through `proximal`:

```python
import jax.numpy as jnp
from proxax import px

# Indefinite (nonconvex) quadratic objective over [-1, 1]^2
q_mat = jnp.array([[3.0, -1.5], [-1.5, -1.0]])
c_vec = jnp.array([-0.8, 0.4])

def quad_fn(x):
  return 0.5 * jnp.dot(x, q_mat @ x) + jnp.dot(c_vec, x)


problem = px.CompositeProblem(
    smooth=quad_fn,
    proximal=px.prox.Box(lower=-1.0, upper=1.0),
)

sol = px.solve(problem=problem, x0=jnp.zeros(2), method="apd", opt_tol=1e-6)
print(sol)
print("x =", sol.x)
```

**Output:**

```text
Solution(
  status=CONVERGED,
  obj=-9.816667e-01,
  opt_res=2.695e-07,
  feas_res=0.000e+00,
  iters=11 (inner=35),
  evals=(smooth=105, grad=80, prox=42),
  runtime_ms=0.36,
  x=float32[2],
  dual=None,
)
x = [-0.23333329 -1.        ]
```

Every iterate — and the final `sol.x` — is guaranteed to lie strictly inside
the box.

---

## 3. Adding Linear Equalities, Inequalities, or Cones

When constraints couple variables through a linear operator $\mathcal{A}(x)$
that does not have a closed-form projection when intersected with `proximal`,
pass `constraints=...` using the helpers in `px.constraints`:

```python
import jax.numpy as jnp
from proxax import px

q_mat = jnp.array([[3.0, -1.5], [-1.5, -1.0]])
c_vec = jnp.array([-0.8, 0.4])

def quad_fn(x):
  return 0.5 * jnp.dot(x, q_mat @ x) + jnp.dot(c_vec, x)


problem = px.CompositeProblem(
    smooth=quad_fn,
    proximal=px.prox.Box(lower=-1.0, upper=1.0),
    constraints=px.constraints.equality(jnp.sum, rhs=0.5),
)

sol = px.solve(
    problem=problem,
    x0=jnp.zeros(2),
    method="ialm",
    opt_tol=1e-5,
    feas_tol=1e-5,
)

print(sol)
print("x =", sol.x)
```

**Output:**

```text
Solution(
  status=CONVERGED,
  obj=-1.352509e-01,
  opt_res=1.251e-06,
  feas_res=3.457e-06,
  iters=9 (inner=222),
  evals=(smooth=550, grad=492, prox=248),
  runtime_ms=0.30,
  x=float32[2],
  dual=float32[],
)
x = [0.29000032 0.21000314]
```

Notice that `px.constraints.equality` accepts either a dense JAX matrix **or**
any linear Python function `x -> A(x)` (such as `jnp.sum`) — Proxax synthesizes
the transpose $\mathcal{A}^*$ automatically via `jax.linear_transpose`.

---

## Understanding the Returned `Solution`

Every solver returns a `px.Solution` PyTree with five fields:

| Field | Type | Plain-English Meaning |
| :--- | :--- | :--- |
| `sol.x` | PyTree | Certified primal solution $\hat{x} \in \operatorname{dom}(h)$ with the exact same PyTree structure as `x0`. |
| `sol.dual` | PyTree or `None` | Lagrange multiplier $\hat{p}$ matching the structure of `problem.constraints` (`None` if unconstrained). |
| `sol.subgrad` | PyTree | Exact stationarity vector $\hat{v} \in \nabla f(\hat{x}) + \partial h(\hat{x}) + \mathcal{A}^*(\hat{p})$. Near zero means $\hat{x}$ is stationary. |
| `sol.status` / `sol.status_enum` | `jax.Array` / `SolveStatus` | Integer code (`0` = `CONVERGED`, `1` = `MAX_ITER_REACHED`, `2` = `TIME_LIMIT_REACHED`, `3` = `LINE_SEARCH_FAILED`, `4` = `NON_FINITE`). |
| `sol.stats` | `SolveStats` | Iteration and oracle counters (`outer_iters`, `inner_iters`, `smooth_evals`, `grad_evals`, `prox_evals`) plus final `opt_residual`, `feas_residual`, and `obj_val`. |

---

## Next Steps

- **Want the mathematical definitions?** Read
  {doc}`foundations/problem_classes` and {doc}`foundations/oracles`.
- **Want plain-English API guides?** Jump to {doc}`guide/solving_problems`,
  {doc}`guide/prox_and_constraints`, and {doc}`guide/optax_and_jax`.
- **Want complete domain benchmarks?** Explore {doc}`examples/unconstrained`,
  {doc}`examples/constrained`, and {doc}`examples/spectral`.
