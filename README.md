# Proxax

[![CI](https://github.com/wwkong/proxax/actions/workflows/ci.yml/badge.svg)](https://github.com/wwkong/proxax/actions/workflows/ci.yml)
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

Minimize an indefinite (nonconvex) quadratic over the unit box with a linear
equality constraint:

$$\min_{x \in [0, 1]^3} \; \frac{1}{2} x^\top Q x + \langle c, x \rangle \quad \text{s.t.} \quad \sum_{i=1}^3 x_i = 1$$

```python
import jax.numpy as jnp
from proxax import px

q_mat = jnp.array([[4.0, -1.5, 0.5], [-1.5, -2.0, 0.3], [0.5, 0.3, 2.5]])
c_vec = jnp.array([-0.6, 0.2, -0.4])


def smooth_fn(x):
  return 0.5 * jnp.dot(x, q_mat @ x) + jnp.dot(c_vec, x)


problem = px.CompositeProblem(
    smooth=smooth_fn,
    proximal=px.prox.Box(lower=0.0, upper=1.0),
    constraints=px.constraints.equality(jnp.sum, rhs=1.0),
)

sol = px.solve(
    problem,
    jnp.ones(3) / 3.0,
    solver=px.solvers.IALM(opt_tol=1e-5, feas_tol=1e-5),
)
print(sol.status_enum.name, sol.x, f"{float(sol.stats.opt_residual):.2e}")
# CONVERGED [0.06000053 0.9399997  0.        ] 6.80e-06
```

## Problem Classes

1. **Unconstrained Composite Optimization (`px.CompositeProblem`):**

   $$\min_{x \in \mathbb{R}^n} \; f(x) + h(x)$$

   where $f$ (`smooth`) is differentiable and $(m, M)$-weakly convex/smooth,
   $h$ (`proximal`) is proper, closed, convex, and proximable, and $x$ can be a
   vector in $\mathbb{R}^n$ or any JAX PyTree of arrays.

2. **Set- and Cone-Constrained Optimization (`px.CompositeProblem`):**

   $$\min_{x \in \mathbb{R}^n} \; f(x) + h(x) \quad \text{s.t.} \quad \mathcal{A}(x) \in \mathcal{C}$$

   where $\mathcal{A}: \mathbb{R}^n \to \mathbb{R}^m$ is a matrix or callable
   PyTree linear map and $\mathcal{C} \subseteq \mathbb{R}^m$ is a closed convex
   set with Euclidean projection $\Pi_{\mathcal{C}}$.

3. **Min-Max & Saddle-Point Optimization (`px.MinMaxProblem`):**

   $$\min_{x \in \mathcal{X}} \max_{y \in \mathcal{Y}} \; \Phi(x, y) + h(x) - g(y)$$

   where $\Phi(x, y)$ (`coupling`) is differentiable and $(h, g)$ (`prox_x`,
   `prox_y`) are proper, closed, convex, and proximable over arbitrary JAX
   PyTrees $(x, y)$.

4. **Matrix Spectral Composite Optimization (`px.SpectralProblem`):**

   $$\min_{X \in \mathbb{R}^{p \times q}} \; f_1(X) + (f_2^{\mathcal{V}} \circ \sigma)(X) + (h^{\mathcal{V}} \circ \sigma)(X)$$

   where $\sigma(X)$ denotes the singular values of $X$ and $(f_2^{\mathcal{V}},
   h^{\mathcal{V}})$ are absolutely symmetric functions on
   $\mathbb{R}^{\min(p, q)}$.

## Available Solvers (`px.solvers`)

*   **Unconstrained Composite (`px.CompositeProblem`):** `APD`, `AIPP`, `ACACG`,
    `AdapFISTA`, `NCFISTA`, `AdapAPG`, `UPFAG`, `AG`, `ECG`
*   **Set / Cone-Constrained (`px.CompositeProblem` with `constraints`):**
    `IALM`, `QPAPD`, `QPAIPP`, `IAIPAL`, `SProxALM`
*   **Min-Max & Saddle-Point (`px.MinMaxProblem`):** `ExtraGradient`, `PDHG`,
    `ProxPointMinMax`
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
  version = {0.0.4},
  year = {2026},
}
```
