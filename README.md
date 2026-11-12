# `proxax`

[![PyPI version](https://img.shields.io/pypi/v/proxax.svg)](https://pypi.org/project/proxax/)
[![Documentation Status](https://readthedocs.org/projects/proxax/badge/?version=latest)](https://proxax.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Certified first-order proximal point methods for unconstrained and constrained
nonconvex composite optimization in **JAX** and **Optax**.

**Documentation:** https://proxax.readthedocs.io/

## Installation

```bash
pip install proxax
```

Or with `uv`:

```bash
uv add proxax
```

## Problem Classes

1. **Unconstrained Nonconvex Composite Optimization:**
   $$\min_{x \in \mathcal{X}} \; f(x) + h(x)$$
   where $f$ (`smooth`) is differentiable and $(m, L^+)$-weakly convex/smooth on
   an arbitrary JAX PyTree $\mathcal{X}$, and $h$ (`proximal`) is proper,
   closed, convex, and proximable.

2. **Set / Cone-Constrained Nonconvex Composite Optimization:**
   $$\min_{x \in \mathcal{X}} \; f(x) + h(x) \quad \text{s.t.} \quad \mathcal{A}(x) \in \mathcal{C}$$
   where $\mathcal{A}: \mathcal{X} \to \mathcal{Y}$ is a linear map between
   PyTrees and $\mathcal{C} \subseteq \mathcal{Y}$ is a closed convex set with
   Euclidean projection $\Pi_{\mathcal{C}}$.

## Quick Example

```python
import jax.numpy as jnp
import proxax
from proxax import constraints
from proxax import prox
from proxax import solvers

problem = proxax.CompositeProblem(
    smooth=lambda x: 0.5 * jnp.dot(x, q_mat @ x) + jnp.dot(c_vec, x),
    proximal=prox.Box(lower=-1.0, upper=1.0),
    constraints=constraints.equality(lambda x: a_mat @ x, rhs=b_vec),
)

sol = proxax.solve(problem, x0, solver=solvers.QPAIPP(opt_tol=1e-5, feas_tol=1e-5))
```

## Available Certified Solvers (`wwkong/nc_opt` Suite)

- **Unconstrained Composite (`CompositeProblem`):** `AIPP`, `APD` (`PF.APD`), `ECG`, `AG`, `UPFAG`, `NCFISTA`, `AdapFISTA`, `ACACG`, `AdapAPG`
- **Unconstrained Spectral (`SpectralProblem` / `NuclearNorm`):** `IAICG`, `DAICG`
- **Set / Cone-Constrained (`CompositeProblem` with `constraints`):** `QPAIPP`, `QPAPD`, `IALM` (`NL-IAPIAL` / `AIDAL`), `SProxALM`, `IAIPAL`
- **Optax Single-Step Transforms (`proxax.optax_compat`):** `proximal`, `aipp`, `apd`, `qp_aipp`, `qp_apd`, `ialm`

## Runnable `nc_opt` Examples

Self-contained benchmark scripts ported from [`wwkong/nc_opt`](https://github.com/wwkong/nc_opt) live in [`examples/`](examples/):

- [`examples/unconstrained_nc_opt.py`](examples/unconstrained_nc_opt.py) — univariate convex QP, nonconvex simplex QP across 7 solvers, and nonconvex `tanh` SVM.
- [`examples/spectral_matrix_completion.py`](examples/spectral_matrix_completion.py) — nonconvex log-sum spectral matrix completion (`IAICG`, `DAICG`, `APD`) vs convex `NuclearNorm`.
- [`examples/constrained_nc_opt.py`](examples/constrained_nc_opt.py) — linearly-constrained nonconvex simplex QP across all 5 constrained solvers, MCP Sparse PCA on the $k$-Fantope (PyTree `IALM`), and box-constrained matrix completion.
- [`examples/optax_and_vmap.py`](examples/optax_and_vmap.py) — single-step trajectory tracking with `proxax.optax_compat` and parallel solving of 16 nonconvex QPs via `jax.jit(jax.vmap(...))`.
