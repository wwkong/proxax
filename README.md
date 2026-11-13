# Proxax

[![PyPI version](https://img.shields.io/pypi/v/proxax.svg)](https://pypi.org/project/proxax/)
[![Documentation Status](https://readthedocs.org/projects/proxax/badge/?version=latest)](https://proxax.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Proxax** is a Python library of **certified first-order proximal, spectral,
penalty, and augmented Lagrangian solvers** for smooth, nonsmooth, and
constrained optimization in **JAX** and **Optax**.

*   **Certified stationarity:** Solvers handle both convex and $(m, M)$-weakly
    convex (nonconvex) objectives and return exact subdifferential and KKT
    certificates $(\hat{x}, \hat{p}, \hat{v})$ rather than heuristic step-size
    checks.
*   **JAX-native & scalable:** Operates directly on arbitrary JAX PyTrees
    (`jax.Array`, nested `dict`/`tuple`, or registered dataclasses), compiles
    end-to-end under `jax.jit`, and vectorizes across batches via `jax.vmap`.
*   **Modular & lightweight:** Clean single-facade API (`from proxax import px`),
    single-step `optax.GradientTransformation` wrappers in `px.optax_compat`,
    and only `jax` and `optax` as runtime dependencies.

[Documentation](https://proxax.readthedocs.io/en/latest/) ·
[User Guide](https://proxax.readthedocs.io/en/latest/guide/solving_problems.html) ·
[API Reference](https://proxax.readthedocs.io/en/latest/api.html) ·
[Roadmap](https://github.com/wwkong/proxax/blob/main/ROADMAP.md) ·
[Open an Issue](https://github.com/wwkong/proxax/issues)

## Installation

```bash
pip install proxax
```

Or with `uv`:

```bash
uv add proxax
```

## Quick Example

```python
import jax.numpy as jnp
from proxax import px

q_mat = jnp.array([[4.0, -1.5, 0.5], [-1.5, -2.0, 0.3], [0.5, 0.3, 2.5]])
c_vec = jnp.array([-0.6, 0.2, -0.4])

problem = px.CompositeProblem(
    smooth=lambda x: 0.5 * jnp.dot(x, q_mat @ x) + jnp.dot(c_vec, x),
    proximal=px.prox.Box(lower=0.0, upper=1.0),
    constraints=px.constraints.equality(lambda x: jnp.sum(x), rhs=1.0),
)

sol = px.solve(
    problem,
    jnp.ones(3) / 3.0,
    solver=px.solvers.IALM(opt_tol=1e-5, feas_tol=1e-5),
)
print(sol.status_enum.name, sol.x, float(sol.stats.opt_residual))
```

## Problem Classes

1. **Unconstrained Composite Optimization (`px.CompositeProblem`):**

   $$\min_{x \in \mathcal{X}} \; f(x) + h(x)$$

   where $f$ (`smooth`) is differentiable and $(m, M)$-weakly convex/smooth on
   an arbitrary JAX PyTree $\mathcal{X}$, and $h$ (`proximal`) is proper,
   closed, convex, and proximable.

2. **Set- and Cone-Constrained Optimization (`px.CompositeProblem`):**

   $$\min_{x \in \mathcal{X}} \; f(x) + h(x) \quad \text{s.t.} \quad \mathcal{A}(x) \in \mathcal{C}$$

   where $\mathcal{A}: \mathcal{X} \to \mathcal{Y}$ is a matrix or callable
   PyTree linear map and $\mathcal{C} \subseteq \mathcal{Y}$ is a closed convex
   set with Euclidean projection $\Pi_{\mathcal{C}}$.

3. **Matrix Spectral Composite Optimization (`px.SpectralProblem`):**

   $$\min_{X \in \mathbb{R}^{p \times q}} \; f_1(X) + (f_2^{\mathcal{V}} \circ \sigma)(X) + (h^{\mathcal{V}} \circ \sigma)(X)$$

   where $\sigma(X)$ denotes the singular values of $X$ and $(f_2^{\mathcal{V}},
   h^{\mathcal{V}})$ are absolutely symmetric functions on
   $\mathbb{R}^{\min(p, q)}$.

## Available Solvers (`px.solvers`)

*   **Unconstrained Composite (`px.CompositeProblem`):** `APD`, `AIPP`, `ACACG`,
    `AdapFISTA`, `NCFISTA`, `AdapAPG`, `UPFAG`, `AG`, `ECG`
*   **Set / Cone-Constrained (`px.CompositeProblem` with `constraints`):**
    `IALM`, `QPAPD`, `QPAIPP`, `IAIPAL`, `SProxALM`
*   **Matrix Spectral (`px.SpectralProblem` / `px.prox.NuclearNorm`):** `DAICG`,
    `IAICG`
*   **Optax Single-Step Transforms (`px.optax_compat`):** `proximal`, `apd`,
    `aipp`, `ialm`, `qp_apd`, `qp_aipp`

## Runnable Examples

Self-contained benchmark scripts live in
[`examples/`](https://github.com/wwkong/proxax/tree/main/examples):

*   [`examples/unconstrained_nc_opt.py`](https://github.com/wwkong/proxax/blob/main/examples/unconstrained_nc_opt.py) — univariate convex QP, nonconvex simplex QP across 7 solvers, and nonconvex `tanh` margin classification.
*   [`examples/constrained_nc_opt.py`](https://github.com/wwkong/proxax/blob/main/examples/constrained_nc_opt.py) — linearly-constrained nonconvex simplex QP across all 5 constrained solvers, MCP Sparse PCA on the $k$-Fantope, and box-constrained matrix completion.
*   [`examples/spectral_matrix_completion.py`](https://github.com/wwkong/proxax/blob/main/examples/spectral_matrix_completion.py) — nonconvex log-sum spectral matrix completion (`DAICG`, `IAICG`, `APD`) vs convex `NuclearNorm`.
*   [`examples/optax_and_vmap.py`](https://github.com/wwkong/proxax/blob/main/examples/optax_and_vmap.py) — single-step trajectory tracking with `px.optax_compat` and parallel solving via `jax.jit(jax.vmap(...))`.

## Citing Proxax

If Proxax was helpful for a publication, please cite this repository:

```bibtex
@software{proxax2026github,
  author = {Weiwei Kong},
  title = {{Proxax}: Certified first-order proximal point methods for composite optimization in JAX and Optax.},
  url = {https://github.com/wwkong/proxax},
  version = {0.0.2},
  year = {2026},
}
```
