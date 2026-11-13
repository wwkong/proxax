# Proxax: Certified Composite Optimization in JAX

**Proxax** is a library of **certified first-order proximal, spectral,
penalty, and augmented Lagrangian solvers** for smooth, nonsmooth, and
constrained optimization in **JAX** and **Optax** — covering both convex and
nonconvex objectives.

Every solver in Proxax:

- **Works natively on JAX PyTrees** (`jax.Array`, nested `dict` / `tuple`, or
  registered dataclasses) without flattening arrays by hand.
- **Compiles end-to-end** under `jax.jit`, vectorizes across problem batches
  with `jax.vmap`, and differentiates through solutions via `px.implicit_solve`.
- **Returns exact stationarity certificates** $(\hat{x}, \hat{p}, \hat{v})$
  measuring true subdifferential and KKT residuals rather than heuristic step
  lengths.
- **Depends only on `jax` and `optax`** at runtime.

---

## Documentation Overview

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Installation & Quickstart
:link: quickstart
:link-type: doc

Install Proxax and solve your first unconstrained, box-constrained, and
linearly-constrained problems in a few lines of JAX.
:::

:::{grid-item-card} Problem Classes & Oracles
:link: foundations/problem_classes
:link-type: doc

Explore the mathematical formulations of composite, spectral, and
cone-constrained problems, curvature bounds $(m, M)$, first-order oracles, and
exact stationarity certificates.
:::

:::{grid-item-card} Plain-English API Guide
:link: guide/solving_problems
:link-type: doc

Learn how to assemble problems, pick solvers, combine `prox` operators and
`constraints` over PyTrees, and step through Optax loops or `jax.vmap` batches.
:::

:::{grid-item-card} Worked Examples
:link: examples/unconstrained
:link-type: doc

Walk through complete runnable benchmarks: simplex QPs, nonconvex SVMs,
spectral matrix completion, $k$-Fantope Sparse PCA, and bounded matrix
completion.
:::

:::{grid-item-card} API Reference
:link: api
:link-type: doc

Browse full class and function signatures across `proxax`, `prox`,
`constraints`, `oracle`, `certificates`, `solvers`, and `optax_compat`.
:::

:::{grid-item-card} Appendices: Theory & `nc_opt` Comparison
:link: appendices/literature_and_theory
:link-type: doc

Read the literature review, worst-case complexity bounds, algorithm deep dives,
and detailed comparisons to the upstream MATLAB `nc_opt` package.
:::

::::

---

## What Problems Can Proxax Solve?

1. **Unconstrained Composite Optimization (`CompositeProblem`):**

   $$\min_{x \in \mathbb{R}^n} \; f(x) + h(x)$$

   Minimize a differentiable (possibly nonconvex) objective $f$ plus a convex
   proximable regularizer or indicator constraint $h$ such as an $\ell_1$ norm,
   elastic net, box, ball, or probability simplex (over $\mathbb{R}^n$ or any
   JAX PyTree of arrays).

2. **Set- and Cone-Constrained Composite Optimization (`CompositeProblem`):**

   $$\min_{x \in \mathbb{R}^n} \; f(x) + h(x) \quad \text{s.t.} \quad \mathcal{A}(x) \in \mathcal{C}$$

   Add linear equalities, inequalities, two-sided box bounds, or second-order
   cone constraints $\mathcal{C} \subseteq \mathbb{R}^m$ over any matrix or
   matrix-free PyTree linear map $\mathcal{A}: \mathbb{R}^n \to \mathbb{R}^m$.

3. **Matrix Spectral Composite Optimization (`SpectralProblem`):**

   $$\min_{X \in \mathbb{R}^{p \times q}} \; f_1(X) + (f_2^{\mathcal{V}} \circ \sigma)(X) + (h^{\mathcal{V}} \circ \sigma)(X)$$

   Minimize a smooth matrix loss $f_1(X)$ coupled with convex or nonconvex
   penalties on the singular values $\sigma(X)$ using a single thin SVD per
   outer iteration.

4. **Min-Max & Saddle-Point Optimization (`MinMaxProblem`):**

   $$\min_{x \in \mathcal{X}} \max_{y \in \mathcal{Y}} \; \Phi(x, y) + h_x(x) - h_y(y)$$

   Solve convex-concave and nonconvex-nonconcave composite saddle-point
   problems using adaptive Mirror-Prox (`ExtraGradient`), Chambolle-Pock
   (`PDHG`), or proximal point (`ProxPointMinMax`).

5. **Block-Constrained Composite Optimization (`BlockProblem`):**

   $$\min_{x_1, \dots, x_K} \; \sum_{i=1}^K \big(f_i(x_i) + h_i(x_i)\big) \quad \text{s.t.} \quad \sum_{i=1}^K \mathcal{A}_i(x_i) = b$$

   Split separable multi-block and consensus problems across coupled
   PyTree blocks using certified proximal `ADMM`.

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
    problem,
    jnp.ones(3) / 3.0,
    solver=px.solvers.IALM(opt_tol=1e-5, feas_tol=1e-5),
)
print(sol.status_enum.name, sol.x, f"{float(sol.stats.opt_residual):.2e}")
# CONVERGED [0.06000053 0.9399997  0.        ] 6.80e-06
```

---

## Citing Proxax

If Proxax was helpful for a publication, please cite this repository:

```bibtex
@software{proxax2026github,
  author = {Weiwei Kong},
  title = {{Proxax}: Certified first-order proximal point methods for composite optimization in JAX and Optax.},
  url = {https://github.com/wwkong/proxax},
  version = {0.2.0},
  year = {2026},
}
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Getting Started

quickstart
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Problem Classes & Oracles

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
guide/plotting
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Worked Examples

examples/unconstrained
examples/constrained
examples/minmax
examples/block
examples/spectral
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Reference

api
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Experiments

experiments/index
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Appendices

appendices/literature_and_theory
appendices/nc_opt_comparison
```
