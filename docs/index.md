# `proxax`: Certified Composite Optimization in JAX

**`proxax`** is a library of **certified first-order proximal point, spectral,
penalty, and augmented Lagrangian methods** for unconstrained and
set/cone-constrained **nonconvex composite optimization** in **JAX** and
**Optax**, porting the complete algorithm suite of
[`wwkong/nc_opt`](https://github.com/wwkong/nc_opt).

Every solver operates natively on arbitrary **JAX PyTrees**, compiles end-to-end
under `jax.jit` and `jax.vmap`, depends strictly on **`jax` and `optax` only**,
and returns **exact primal-dual stationarity certificates** rather than
unverified raw iterates.

---

## Explore the Documentation

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Getting Started & Solver Guide
:link: quickstart
:link-type: doc

Install `proxax`, understand exact stationarity certificates $(\hat{x}, \hat{v})$,
build `CompositeProblem` PyTrees, and pick the right solver across all 16
methods.
:::

:::{grid-item-card} Unconstrained `nc_opt` Examples
:link: examples/unconstrained
:link-type: doc

Solve univariate convex QPs, simplex-constrained nonconvex QPs, and nonconvex
$\tanh$-loss SVMs with parameter-free `APD`, `AIPP`, `ACACG`, `AdapFISTA`, and
`UPFAG`.
:::

:::{grid-item-card} Spectral Matrix Completion
:link: examples/spectral
:link-type: doc

Exploit thin-SVD structure in nonconvex log-sum and nuclear-norm matrix
completion with `SpectralProblem`, `IAICG`, and doubly-accelerated `DAICG`.
:::

:::{grid-item-card} Constrained Penalty & ALM Examples
:link: examples/constrained
:link-type: doc

Tackle linearly-constrained simplex QPs, $k$-Fantope consensus Sparse PCA, and
bounded matrix completion with `QPAPD`, `QPAIPP`, `IALM`, `IAIPAL`, and
`SProxALM`.
:::

:::{grid-item-card} Optax Transforms & `jax.vmap`
:link: examples/optax_and_jax
:link-type: doc

Step through certified proximal stages inside custom `optax` loops and solve
batches of nonconvex programs in parallel on GPU with `jax.jit(jax.vmap(...))`.
:::

:::{grid-item-card} API & Theory Reference
:link: api
:link-type: doc

Browse full signatures for `CompositeProblem`, `SpectralProblem`, `prox`,
`constraints`, all 16 certified solvers, and the `nc_opt` / `PF.APD` theory map.
:::

::::

---

## Problem Classes at a Glance

1. **Unconstrained Nonconvex Composite Optimization (`CompositeProblem`):**

   $$\min_{x \in \mathcal{X}} \; f(x) + h(x)$$

   where $f$ is differentiable and $(m, M)$-weakly convex/smooth on a PyTree
   space $\mathcal{X}$, and $h$ is proper, closed, convex, and proximable.

2. **Unconstrained Matrix Spectral Optimization (`SpectralProblem`):**

   $$\min_{X \in \mathbb{R}^{p \times q}} \; f_1(X) + (f_2^{\mathcal{V}} \circ \sigma)(X) + (h^{\mathcal{V}} \circ \sigma)(X)$$

   where $\sigma(X)$ denotes the singular values of $X$ and $(f_2^{\mathcal{V}},
   h^{\mathcal{V}})$ are absolutely symmetric functions on $\mathbb{R}^{\min(p, q)}$.

3. **Set / Cone-Constrained Nonconvex Composite Optimization:**

   $$\min_{x \in \mathcal{X}} \; f(x) + h(x) \quad \text{s.t.} \quad \mathcal{A}(x) \in \mathcal{C}$$

   where $\mathcal{A}: \mathcal{X} \to \mathcal{Y}$ is a matrix or PyTree linear
   map (with automatic `jax.linear_transpose` adjoint $\mathcal{A}^*$) and
   $\mathcal{C} \subseteq \mathcal{Y}$ is a closed convex set with Euclidean
   projector $\Pi_{\mathcal{C}}$.

---

## 30-Second Quick Example

```python
import jax.numpy as jnp
import proxax
from proxax import constraints
from proxax import prox
from proxax import solvers

q_mat = jnp.array([[4.0, -1.5, 0.5], [-1.5, -2.0, 0.3], [0.5, 0.3, 2.5]])
c_vec = jnp.array([-0.6, 0.2, -0.4])
a_mat = jnp.array([[1.0, 1.0, 1.0]])
b_vec = jnp.array([1.0])

problem = proxax.CompositeProblem(
    smooth=lambda x: 0.5 * jnp.dot(x, q_mat @ x) + jnp.dot(c_vec, x),
    proximal=prox.Box(lower=0.0, upper=1.0),
    constraints=constraints.equality(lambda x: a_mat @ x, rhs=b_vec),
)

sol = proxax.solve(
    problem,
    jnp.ones(3) / 3.0,
    solver=solvers.IALM(opt_tol=1e-5, feas_tol=1e-5),
)
print(sol.status_enum.name, sol.x, float(sol.stats.opt_residual))
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
:caption: Examples (from nc_opt)

examples/unconstrained
examples/spectral
examples/constrained
examples/optax_and_jax
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Reference & Theory

nc_opt_reference
api
```
