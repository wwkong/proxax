# Problems, Solvers & `proxax.solve`

This guide walks through the day-to-day workflow of `proxax` in plain English:
building problems over arrays or nested PyTrees, picking a solver, tuning
tolerances, logging progress, and inspecting solutions.

---

## 1. Building a Problem

### `CompositeProblem` (Most Problems)

Use `proxax.CompositeProblem` whenever you want to minimize a smooth function
`smooth`, a nonsmooth regularizer or simple set `proximal`, and/or linear/cone
constraints `constraints`:

```python
import jax.numpy as jnp
import proxax
from proxax import constraints
from proxax import prox

problem = proxax.CompositeProblem(
    smooth=lambda x: 0.5 * jnp.sum((x - 1.5) ** 2),
    proximal=prox.L1(weight=0.2),
    constraints=None,  # omit or pass a constraint from `proxax.constraints`
)
```

All three arguments are optional:

- Omit `smooth` (or pass `None`) for pure feasibility / projection problems.
- Omit `proximal` for smooth unconstrained or smooth linearly-constrained
  problems.
- Omit `constraints` when there is no coupling constraint $\mathcal{A}(x) \in \mathcal{C}$.

### Using PyTree Variables (`dict`, `tuple`, Dataclasses)

Your optimization variable `x` does **not** have to be a flat 1D vector. Any
JAX PyTree works out of the box:

```python
import jax
import jax.numpy as jnp
import proxax
from proxax import prox

features = jnp.array([[1.0, 2.0], [0.5, -1.0], [-1.5, 0.5]])
targets = jnp.array([1.2, -0.4, -0.8])

def regression_loss(params: dict[str, jax.Array]) -> jax.Array:
  preds = features @ params["w"] + params["b"]
  return 0.5 * jnp.mean((preds - targets) ** 2)

# Penalize weights `w` with L1, leave scalar bias `b` unpenalized:
problem = proxax.CompositeProblem(
    smooth=regression_loss,
    proximal=prox.MultiProx({"w": prox.L1(weight=0.1), "b": prox.Zero()}),
)

x0 = {"w": jnp.zeros(2), "b": jnp.asarray(0.0)}
sol = proxax.solve(problem, x0, method="apd")
print(sol.x["w"], sol.x["b"])
```

### `SpectralProblem` (Matrix Singular-Value Problems)

When optimizing over a 2D matrix $X \in \mathbb{R}^{p \times q}$ with penalties
that depend only on the singular values $\sigma(X)$, use `proxax.SpectralProblem`
so spectral solvers (`DAICG`, `IAICG`) only compute **one thin SVD per outer
iteration**:

```python
import jax.numpy as jnp
import proxax
from proxax import prox

target = jnp.array([[3.0, 0.0], [0.0, 0.5], [0.0, 0.0]])

spec_prob = proxax.SpectralProblem(
    f1=lambda x_mat: 0.5 * jnp.sum((x_mat - target) ** 2),
    f2_vec=lambda s: -0.1 * jnp.sum(s**2 / (1.0 + s**2)),  # smooth nonconvex on singular values
    prox_vec=prox.L1(weight=0.25),                         # 1D L1 on singular values = nuclear norm
)

sol = proxax.solve(spec_prob, jnp.zeros((3, 2)), method="da_icg", opt_tol=1e-5)
```

---

## 2. Calling `proxax.solve`

You can invoke `proxax.solve` in two ways:

::::{tab-set}

:::{tab-item} Quick String Name (`method=...`)
Pass `method="..."` along with `opt_tol` and `feas_tol` to use sensible default
hyperparameters:

```python
sol = proxax.solve(problem, x0, method="apd", opt_tol=1e-5)
```
:::

:::{tab-item} Configured Solver Object (`solver=...`)
Instantiate any class from `proxax.solvers` when you want fine-grained control
over iteration budgets, norm kinds, or penalty schedules:

```python
from proxax import solvers

solver = solvers.APD(
    opt_tol=1e-6,
    norm_kind="linf",
    max_iters=500,
    max_inner_iters=1000,
)
sol = proxax.solve(problem, x0, solver=solver)
```
:::

::::

---

## 3. Which Solver Should I Pick?

If you just want a fast, reliable answer without tuning anything, start with
these three defaults:

| Problem Type | Recommended Default | `method=` String | Why Start Here? |
| :--- | :--- | :--- | :--- |
| **Unconstrained** (`constraints=None`) | `solvers.APD` | `"apd"` | Completely parameter-free; automatically adapts to both convex and nonconvex regions. |
| **Matrix Spectral** (`SpectralProblem` or `NuclearNorm`) | `solvers.DAICG` | `"da_icg"` | Doubly accelerated; requires only one thin SVD per outer iteration. |
| **Constrained** (`constraints=...`) | `solvers.IALM` or `solvers.QPAPD` | `"ialm"` / `"qp_apd"` | Both are parameter-free out of the box; `IALM` updates dual multipliers for moderate penalties, while `QPAPD` uses pure quadratic penalties. |

### Full Solver Catalog at a Glance

- **Unconstrained solvers (`proxax.solvers`):**
  - **Parameter-free (no curvature constants needed):** `APD` (`"apd"`),
    `AdapFISTA` (`"adap_fista"`), `AdapAPG` (`"adap_apg"`), `UPFAG` (`"upfag"`),
    `ACACG` (`"ac_acg"`), and `ECG` (`"ecg"`, simple backtracking proximal
    gradient).
  - **Curvature-guided (fastest when `SmoothTerm` supplies `lower_curvature` /
    `upper_curvature`):** `AIPP` (`"aipp"`), `NCFISTA` (`"nc_fista"`), and `AG`
    (`"ag"`).
- **Matrix spectral solvers:** `DAICG` (`"da_icg"`) and `IAICG` (`"ia_icg"`).
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

For theoretical complexity bounds and literature references behind each method,
see {doc}`../appendices/literature_and_theory`.

---

## 4. Compilation, Callbacks & Time Limits (`jit=True` vs `jit=False`)

By default, `proxax.solve(..., jit=True)` compiles the entire solver loop with
`jax.lax.while_loop`. This gives minimal Python overhead on CPU/GPU and makes
every solver compatible with `jax.jit` and `jax.vmap`.

When debugging interactively in Python — or when you want a **per-iteration
callback** or a **wall-clock `time_limit`** (in seconds) — pass `jit=False`:

```python
import proxax
from proxax import solvers

def log_outer_step(info: dict[str, object]) -> None:
  print(info)

sol = proxax.solve(
    problem,
    x0,
    solver=solvers.APD(opt_tol=1e-6),
    jit=False,
    callback=log_outer_step,
    time_limit=5.0,
)
```

---

## 5. Inspecting Solutions & Verifying KKT Residuals

Every `Solution` object exposes diagnostic counters and an independent KKT check:

```python
from proxax import SolveStatus

assert sol.status_enum == SolveStatus.CONVERGED

print("Objective value:      ", float(sol.stats.obj_val))
print("Stationarity ||v||:   ", float(sol.stats.opt_residual))
print("Feasibility residual: ", float(sol.stats.feas_residual))
print("Outer / inner iters:  ", int(sol.stats.outer_iters), int(sol.stats.inner_iters))
print("Grad / prox evals:    ", int(sol.stats.grad_evals), int(sol.stats.prox_evals))

# Independently re-evaluate KKT residuals on (sol.x, sol.subgrad):
kkt = problem.evaluate_residuals(sol.x, sol.subgrad, norm_kind="l2")
print("Verified stationarity:", float(kkt.stationarity_norm))
```
