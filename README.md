# `proxax`

[![PyPI version](https://img.shields.io/pypi/v/proxax.svg)](https://pypi.org/project/proxax/)
[![Documentation Status](https://readthedocs.org/projects/proxax/badge/?version=latest)](https://proxax.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Certified first-order proximal, spectral, penalty, and augmented Lagrangian
solvers for smooth, nonsmooth, and constrained optimization in **JAX** and
**Optax**.

**Documentation:** https://proxax.readthedocs.io/en/latest/

## Installation

```bash
pip install proxax
```

Or with `uv`:

```bash
uv add proxax
```

## Problem Classes

1. **Unconstrained Composite Optimization (`CompositeProblem`):**
   $$\min_{x \in \mathcal{X}} \; f(x) + h(x)$$
   where $f$ (`smooth`) is differentiable and $(m, M)$-weakly convex/smooth on
   an arbitrary JAX PyTree $\mathcal{X}$, and $h$ (`proximal`) is proper,
   closed, convex, and proximable.

2. **Matrix Spectral Composite Optimization (`SpectralProblem`):**
   $$\min_{X \in \mathbb{R}^{p \times q}} \; f_1(X) + (f_2^{\mathcal{V}} \circ \sigma)(X) + (h^{\mathcal{V}} \circ \sigma)(X)$$
   where $\sigma(X)$ denotes the singular values of $X$ and $(f_2^{\mathcal{V}},
   h^{\mathcal{V}})$ are absolutely symmetric functions on
   $\mathbb{R}^{\min(p, q)}$.

3. **Set- and Cone-Constrained Optimization (`CompositeProblem`):**
   $$\min_{x \in \mathcal{X}} \; f(x) + h(x) \quad \text{s.t.} \quad \mathcal{A}(x) \in \mathcal{C}$$
   where $\mathcal{A}: \mathcal{X} \to \mathcal{Y}$ is a matrix or callable
   PyTree linear map and $\mathcal{C} \subseteq \mathcal{Y}$ is a closed convex
   set with Euclidean projection $\Pi_{\mathcal{C}}$.

## Quick Example

```python
import jax.numpy as jnp
import proxax
from proxax import constraints
from proxax import prox

q_mat = jnp.array([[4.0, -1.5, 0.5], [-1.5, -2.0, 0.3], [0.5, 0.3, 2.5]])
c_vec = jnp.array([-0.6, 0.2, -0.4])

problem = proxax.CompositeProblem(
    smooth=lambda x: 0.5 * jnp.dot(x, q_mat @ x) + jnp.dot(c_vec, x),
    proximal=prox.Box(lower=0.0, upper=1.0),
    constraints=constraints.equality(lambda x: jnp.sum(x), rhs=1.0),
)

sol = proxax.solve(problem, jnp.ones(3) / 3.0, method="ialm", opt_tol=1e-5, feas_tol=1e-5)
print(sol.status_enum.name, sol.x, float(sol.stats.opt_residual))
```

## Available Solvers

- **Unconstrained Composite (`CompositeProblem`):** `APD`, `AIPP`, `ACACG`,
  `AdapFISTA`, `NCFISTA`, `AdapAPG`, `UPFAG`, `AG`, `ECG`
- **Matrix Spectral (`SpectralProblem` / `NuclearNorm`):** `DAICG`, `IAICG`
- **Set / Cone-Constrained (`CompositeProblem` with `constraints`):** `IALM`,
  `QPAPD`, `QPAIPP`, `IAIPAL`, `SProxALM`
- **Optax Single-Step Transforms (`proxax.optax_compat`):** `proximal`, `apd`,
  `aipp`, `ialm`, `qp_apd`, `qp_aipp`

## Runnable Examples

Self-contained benchmark scripts live in [`examples/`](examples/):

- [`examples/unconstrained_nc_opt.py`](examples/unconstrained_nc_opt.py) — univariate convex QP, nonconvex simplex QP across 7 solvers, and nonconvex `tanh` margin classification.
- [`examples/spectral_matrix_completion.py`](examples/spectral_matrix_completion.py) — nonconvex log-sum spectral matrix completion (`DAICG`, `IAICG`, `APD`) vs convex `NuclearNorm`.
- [`examples/constrained_nc_opt.py`](examples/constrained_nc_opt.py) — linearly-constrained nonconvex simplex QP across all 5 constrained solvers, MCP Sparse PCA on the $k$-Fantope, and box-constrained matrix completion.
- [`examples/optax_and_vmap.py`](examples/optax_and_vmap.py) — single-step trajectory tracking with `proxax.optax_compat` and parallel solving via `jax.jit(jax.vmap(...))`.
