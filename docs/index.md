# Proxax

**Certified Composite Optimization in JAX**

[![PyPI](https://img.shields.io/pypi/v/proxax)](https://pypi.org/project/proxax/)
[![Python](https://img.shields.io/pypi/pyversions/proxax)](https://pypi.org/project/proxax/)
[![CI](https://github.com/wwkong/proxax/actions/workflows/ci.yml/badge.svg)](https://github.com/wwkong/proxax/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/readthedocs/proxax)](https://proxax.readthedocs.io/en/latest/)

**Proxax** is a library of **certified first-order proximal, spectral,
penalty, and augmented Lagrangian solvers** for smooth, nonsmooth, and
constrained optimization in **JAX** and **Optax** — covering both convex and
nonconvex objectives.

## Features

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} PyTree-Native
Works natively on arbitrary JAX PyTrees (`jax.Array`, nested `dict` / `tuple`,
or registered dataclasses) without flattening arrays by hand.
:::

:::{grid-item-card} JIT, `vmap` & Differentiable
Compiles end-to-end under `jax.jit`, vectorizes across problem batches with
`jax.vmap`, and differentiates through solutions via `px.implicit_solve`.
:::

:::{grid-item-card} Certified Stationarity
Returns exact stationarity certificates $(\hat{x}, \hat{p}, \hat{v})$ measuring
true subdifferential and KKT residuals rather than heuristic step lengths.
:::

:::{grid-item-card} Pure JAX & Optax
Depends only on `jax` and `optax` at runtime, with first-class Optax optimizer
interoperability.
:::

::::

---

## 30-Second Example

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
    problem=problem,
    x0=jnp.ones(3) / 3.0,
    solver=px.solvers.IALM(opt_tol=1e-5, feas_tol=1e-5),
)
print(sol)
print("x =", sol.x)
```

**Output:**

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

Proxax supports unconstrained, set- and cone-constrained, matrix spectral, min-max, and multi-block composite problems — see [Problem Classes & Oracles](foundations/problem_classes.md) for the full formulations.

---

## Installation

```bash
pip install proxax
```

See [Installation & Quickstart](quickstart.md) for `uv`, GPU/CUDA, and optional extras.

---

## Citing Proxax

If Proxax was helpful for a publication, please cite this repository:

```bibtex
@software{proxax2026github,
  author = {Weiwei Kong},
  title = {{Proxax}: Certified first-order proximal point methods for composite optimization in JAX and Optax.},
  url = {https://github.com/wwkong/proxax},
  version = {0.3.6},
  year = {2026},
}
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Getting Started

quickstart
foundations/problem_classes
foundations/oracles
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: User Guide

guide/solving_problems
guide/prox_and_constraints
guide/optax_and_jax
guide/batching_and_parallelism
guide/implicit_diff
guide/plotting
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Reference & Resources

examples/index
api
Experiments <https://proxax.readthedocs.io/projects/experiments/en/latest/>
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Appendices

Appendix A: Literature & Theory <appendices/literature_and_theory>
appendices/nc_opt_comparison
appendices/oracles_and_certificates
appendices/implementation_optimizations
```
