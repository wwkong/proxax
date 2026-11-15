<p align="center">
  <img src="https://raw.githubusercontent.com/wwkong/proxax/main/docs/_static/logo.png" alt="Proxax logo" width="180" />
</p>

# Proxax

[![CI](https://github.com/wwkong/proxax/actions/workflows/ci.yml/badge.svg)](https://github.com/wwkong/proxax/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/proxax.svg)](https://pypi.org/project/proxax/)
[![Documentation Status](https://img.shields.io/readthedocs/proxax)](https://proxax.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Proxax** is a pure-JAX/Optax library of **certified first-order solvers** for
convex and $(m, M)$-weakly convex composite, constrained, spectral,
multi-block, and min–max optimization.

* **Certified KKT stationarity:** Refinement-based subdifferential and feasibility checks across all problem classes.
* **JAX-native & scalable:** Arbitrary PyTrees, `jax.jit`, sparse `BCOO` & matrix-free linear operators, `px.batch_solve` (`threads` / `vmap`), and differentiable layers (`px.implicit_solve`, `px.layers`).
* **Lightweight:** Clean facade (`from proxax import px`), `optax.GradientTransformation` wrappers (`px.optax_compat`), and only `jax` + `optax` as runtime dependencies.

[Documentation](https://proxax.readthedocs.io/en/latest/) ·
[Quickstart](https://proxax.readthedocs.io/en/latest/quickstart.html) ·
[User Guide](https://proxax.readthedocs.io/en/latest/guide/solving_problems.html) ·
[API Reference](https://proxax.readthedocs.io/en/latest/api.html) ·
[Changelog](https://github.com/wwkong/proxax/blob/main/CHANGELOG.md)

## Installation

```bash
pip install proxax   # or: uv add proxax
```

## Quick Example

Minimize a nonconvex quadratic over $[0, 1]^3$ subject to $\sum_{i=1}^3 x_i = 1$:

```python
import jax.numpy as jnp
from proxax import px

q_mat = jnp.array([[4.0, -1.5, 0.5], [-1.5, -2.0, 0.3], [0.5, 0.3, 2.5]])
c_vec = jnp.array([-0.6, 0.2, -0.4])

problem = px.CompositeProblem(
    smooth=lambda x: 0.5 * jnp.dot(x, q_mat @ x) + jnp.dot(c_vec, x),
    proximal=px.prox.Box(lower=0.0, upper=1.0),
    constraints=px.constraints.equality(jnp.sum, rhs=1.0),
)

sol = px.solve(
    problem=problem,
    x0=jnp.ones(3) / 3.0,
    solver=px.solvers.IALM(opt_tol=1e-5, feas_tol=1e-5),
)
print(sol)
print("x =", sol.x)
```

```text
Solution(
  status=CONVERGED,
  obj=-8.090005e-01,
  opt_res=5.972e-06,
  feas_res=2.384e-07,
  iters=8 (inner=423),
  evals=(smooth=991, grad=892, prox=449),
  runtime_ms=0.46,
  x=float32[3],
  dual=float32[],
)
x = [0.06000032 0.93999994 0.        ]
```

## Problem Classes & Solvers

| Problem Class | Container | Representative Solvers |
| :--- | :--- | :--- |
| Unconstrained composite | `px.CompositeProblem` | `PANOC`, `APD`, `AIPP`, `ACACG`, `AdapFISTA`, `NCFISTA` |
| Set- & cone-constrained | `px.CompositeProblem` | `IALM` (`PANOC-ALM`), `IAIPAL`, `SProxALM`, `QPAPD`, `QPAIPP` |
| Matrix spectral | `px.SpectralProblem` | `DAICG`, `IAICG` |
| Multi-block splitting | `px.BlockProblem` | `ADMM`, `DPADMM`, `PAdaptADMM` |
| Min–max saddle point | `px.MinMaxProblem` | `PDHG`, `ExtraGradient`, `ProxPointMinMax` |

Plus single-step Optax wrappers in `px.optax_compat`, Modified Ruiz Equilibration (`px.ruiz_equilibrate`), and differentiable layers (`px.layers`, `px.implicit_solve`). See [Problem Classes](https://proxax.readthedocs.io/en/latest/foundations/problem_classes.html) and the [API Reference](https://proxax.readthedocs.io/en/latest/api.html) for the full list of 23 solver classes.

## Examples

Runnable scripts live in [`examples/`](https://github.com/wwkong/proxax/tree/main/examples):

* [`constrained.py`](https://github.com/wwkong/proxax/blob/main/examples/constrained.py) — linearly constrained nonconvex QP, sparse PCA on the $k$-Fantope, and matrix completion.
* [`batch_solve_and_threads.py`](https://github.com/wwkong/proxax/blob/main/examples/batch_solve_and_threads.py) — parallel CPU threads vs. `jax.vmap` batching.
* [`layers_training.py`](https://github.com/wwkong/proxax/blob/main/examples/layers_training.py) — end-to-end training through `px.layers` and `px.implicit_solve`.
* [`optax_and_vmap.py`](https://github.com/wwkong/proxax/blob/main/examples/optax_and_vmap.py) — single-step `px.optax_compat` trajectories and `vmap` sweeps.

## Citing Proxax

```bibtex
@software{proxax2026github,
  author = {Weiwei Kong},
  title = {{Proxax}: Certified first-order proximal point methods for composite optimization in JAX and Optax.},
  url = {https://github.com/wwkong/proxax},
  version = {0.4.1},
  year = {2026},
}
```
