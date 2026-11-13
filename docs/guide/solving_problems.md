# Problems, Solvers & `px.solve`

This guide walks through the day-to-day workflow of Proxax in plain English:
importing the unified `px` namespace, building problems over arrays or nested
PyTrees, picking a solver, tuning tolerances, logging progress, and inspecting
solutions.

---

## 1. Unified Namespace (`from proxax import px`)

You only need a single import line (`from proxax import px`, or equivalently
`import proxax as px`) to access the entire library:

- **Core entry points:** `px.CompositeProblem`, `px.SpectralProblem`,
  `px.solve`, `px.Solution`, `px.SolveStatus`, `px.StoppingConfig`
- **Qualified submodules:** `px.prox`, `px.constraints`, `px.solvers`,
  `px.oracle`, `px.optax_compat`, `px.certificates`, `px.tree`, `px.typing`

---

## 2. Building a Problem

### `px.CompositeProblem` (Most Problems)

Use `px.CompositeProblem` whenever you want to minimize a smooth function
`smooth`, a nonsmooth regularizer or simple set `proximal`, and/or linear/cone
constraints `constraints`. For example, to solve the L1-regularized
least-squares problem

$$\min_{x \in \mathbb{R}^n} \; \frac{1}{2}\|x - 1.5\|_2^2 + 0.2 \|x\|_1$$

write:

```python
import jax.numpy as jnp
from proxax import px

def smooth_fn(x):
  return 0.5 * jnp.sum((x - 1.5) ** 2)


problem = px.CompositeProblem(
    smooth=smooth_fn,
    proximal=px.prox.L1(weight=0.2),
    constraints=None,  # omit or pass a constraint from `px.constraints`
)
```

All three arguments are optional:

- Omit `smooth` (or pass `None`) for pure feasibility / projection problems.
- Omit `proximal` for smooth unconstrained or smooth linearly-constrained
  problems.
- Omit `constraints` when there is no coupling constraint $\mathcal{A}(x) \in \mathcal{C}$.

### Using PyTree Variables (`dict`, `tuple`, Dataclasses)

Your optimization variable `x` does **not** have to be a flat 1D vector. Any
JAX PyTree works out of the box—for instance, sparse linear regression with
weights $w \in \mathbb{R}^2$ and unpenalized scalar bias $b \in \mathbb{R}$:

$$\min_{w \in \mathbb{R}^2,\; b \in \mathbb{R}} \; \frac{1}{2N}\|X w + b \mathbf{1} - y\|_2^2 + 0.1\|w\|_1$$

```python
import jax
import jax.numpy as jnp
from proxax import px

features = jnp.array([[1.0, 2.0], [0.5, -1.0], [-1.5, 0.5]])
targets = jnp.array([1.2, -0.4, -0.8])

def regression_loss(params: dict[str, jax.Array]) -> jax.Array:
  preds = features @ params["w"] + params["b"]
  return 0.5 * jnp.mean((preds - targets) ** 2)

# Penalize weights `w` with L1, leave scalar bias `b` unpenalized:
problem = px.CompositeProblem(
    smooth=regression_loss,
    proximal=px.prox.MultiProx({"w": px.prox.L1(weight=0.1), "b": px.prox.Zero()}),
)

x0 = {"w": jnp.zeros(2), "b": jnp.asarray(0.0)}
sol = px.solve(problem, x0, method="apd")
print(sol.x["w"], sol.x["b"])
```

### `px.SpectralProblem` (Matrix Singular-Value Problems)

When optimizing over a 2D matrix $X \in \mathbb{R}^{p \times q}$ with penalties
that depend only on the singular values $\sigma(X)$, use `px.SpectralProblem`
so spectral solvers (`DAICG`, `IAICG`) only compute **one thin SVD per outer
iteration**:

$$\min_{X \in \mathbb{R}^{3 \times 2}} \; \frac{1}{2}\|X - T\|_F^2 - 0.1 \sum_{i=1}^2 \frac{\sigma_i(X)^2}{1 + \sigma_i(X)^2} + 0.25 \|X\|_*$$

```python
import jax.numpy as jnp
from proxax import px

target = jnp.array([[3.0, 0.0], [0.0, 0.5], [0.0, 0.0]])

def f1_fn(x_mat):
  return 0.5 * jnp.sum((x_mat - target) ** 2)


def f2_vec_fn(s):
  return -0.1 * jnp.sum(s**2 / (1.0 + s**2))  # smooth nonconvex on singular values


spec_prob = px.SpectralProblem(
    f1=f1_fn,
    f2_vec=f2_vec_fn,
    prox_vec=px.prox.L1(weight=0.25),  # 1D L1 on singular values = nuclear norm
)

sol = px.solve(spec_prob, jnp.zeros((3, 2)), method="da_icg", opt_tol=1e-5)
```

---

## 3. Calling `px.solve`

You can invoke `px.solve` in two ways:

::::{tab-set}

:::{tab-item} Quick String Name (`method=...`)
Pass `method="..."` along with `opt_tol` and `feas_tol` to use sensible default
hyperparameters:

```python
sol = px.solve(problem, x0, method="apd", opt_tol=1e-5)
```
:::

:::{tab-item} Configured Solver Object (`solver=...`)
Instantiate any class from `px.solvers` when you want fine-grained control over
iteration budgets, norm kinds, or penalty schedules:

```python
solver = px.solvers.APD(
    opt_tol=1e-6,
    norm_kind="linf",
    max_iters=500,
    max_inner_iters=1000,
)
sol = px.solve(problem, x0, solver=solver)
```
:::

::::

---

## 4. Which Solver Should I Pick?

If you just want a fast, reliable answer without tuning anything, start with
these three defaults:

| Problem Type | Recommended Default | `method=` String | Why Start Here? |
| :--- | :--- | :--- | :--- |
| **Unconstrained** (`constraints=None`) | `px.solvers.APD` | `"apd"` | Completely parameter-free; automatically adapts to both convex and nonconvex regions. |
| **Constrained** (`constraints=...`) | `px.solvers.IALM` or `px.solvers.QPAPD` | `"ialm"` / `"qp_apd"` | Both are parameter-free out of the box; `IALM` updates dual multipliers for moderate penalties, while `QPAPD` uses pure quadratic penalties. |
| **Matrix Spectral** (`SpectralProblem` or `NuclearNorm`) | `px.solvers.DAICG` | `"da_icg"` | Doubly accelerated; requires only one thin SVD per outer iteration. |

### Full Solver Catalog at a Glance (`px.solvers`)

- **Unconstrained solvers:**
  - **Parameter-free (no curvature constants needed):** `APD` (`"apd"`),
    `AdapFISTA` (`"adap_fista"`), `AdapAPG` (`"adap_apg"`), `UPFAG` (`"upfag"`),
    `ACACG` (`"ac_acg"`), and `ECG` (`"ecg"`, simple backtracking proximal
    gradient).
  - **Curvature-guided (fastest when `px.oracle.SmoothTerm` supplies
    `lower_curvature` / `upper_curvature`):** `AIPP` (`"aipp"`), `NCFISTA`
    (`"nc_fista"`), and `AG` (`"ag"`).
- **Set/cone-constrained solvers:**
  - `IALM` (`"ialm"`): Inexact proximal augmented Lagrangian; supports `APD` or
    `AIPP` inner solvers and optional multiplier dampening (`dampen` in
    $[0, 1)$, also available via preset `method="aidal"`).
  - `QPAPD` (`"qp_apd"`) & `QPAIPP` (`"qp_aipp"`): Quadratic penalty methods
    driven by `APD` or `AIPP`.
  - `IAIPAL` (`"iaipal"`): Classical proximal augmented Lagrangian taking a
    single inner `ACG` step per outer iteration when curvatures are known.
  - `SProxALM` (`"s_prox_alm"`): Single-loop smoothed proximal ALM for linear
    equality and box/polyhedral constraints.
- **Matrix spectral solvers:** `DAICG` (`"da_icg"`) and `IAICG` (`"ia_icg"`).

For theoretical complexity bounds and literature references behind each method,
see {doc}`../appendices/literature_and_theory`.

---

## 5. Compilation, Callbacks & Time Limits (`jit=True` vs `jit=False`)

By default, `px.solve(..., jit=True)` compiles the entire solver loop with
`jax.lax.while_loop`. This gives minimal Python overhead on CPU/GPU and makes
every solver compatible with `jax.jit` and `jax.vmap`.

When debugging interactively in Python — or when you want a **per-iteration
callback** or a **wall-clock `time_limit`** (in seconds) — pass `jit=False`:

```python
from proxax import px

def log_outer_step(info: object) -> None:
  print(info)

sol = px.solve(
    problem,
    x0,
    solver=px.solvers.APD(opt_tol=1e-6),
    jit=False,
    callback=log_outer_step,
    time_limit=5.0,
)
```

---

## 6. Inspecting Solutions & Verifying KKT Residuals

Every `px.Solution` object exposes diagnostic counters and an independent KKT
check:

```python
assert sol.status_enum == px.SolveStatus.CONVERGED

print("Objective value:      ", float(sol.stats.obj_val))
print("Stationarity ||v||:   ", float(sol.stats.opt_residual))
print("Feasibility residual: ", float(sol.stats.feas_residual))
print("Outer / inner iters:  ", int(sol.stats.outer_iters), int(sol.stats.inner_iters))
print("Grad / prox evals:    ", int(sol.stats.grad_evals), int(sol.stats.prox_evals))

# Independently re-evaluate KKT residuals on (sol.x, sol.subgrad):
kkt = problem.evaluate_residuals(sol.x, sol.subgrad, norm_kind="l2")
print("Verified stationarity:", float(kkt.stationarity_norm))
```
