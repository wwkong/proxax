# Multi-Block ADMM Splitting

This page demonstrates linearly-coupled multi-block optimization in Proxax using
`px.BlockProblem` and `px.BlockTerm`:

$$\min_{x_1, \dots, x_K} \; \sum_{i=1}^K \big(f_i(x_i) + h_i(x_i)\big) \quad \text{s.t.} \quad \sum_{i=1}^K \mathcal{A}_i(x_i) = b$$

All three block solvers (`ADMM`, `DPADMM`, and `PAdaptADMM`) return exact
primal-dual KKT stationarity and feasibility certificates.

All code on this page can be run directly via `examples/block.py`:

```bash
uv run python examples/block.py
```

---

## 1. Two-Block Lasso Consensus Splitting

Splitting a Lasso problem $\min_x \frac{1}{2}\|Ax - b\|_2^2 + \lambda\|z\|_1$
subject to $x - z = 0$ separates the smooth least-squares term from the
nonsmooth $\ell_1$ regularizer:

```python
import jax.numpy as jnp
from proxax import px

a_mat = jnp.array([[2.0, 0.5, -0.3], [0.0, 1.5, 0.4], [-0.2, 0.3, 1.8]])
b_vec = jnp.array([1.0, -0.5, 0.8])

block1 = px.BlockTerm(
    smooth=lambda x: 0.5 * jnp.sum((a_mat @ x - b_vec) ** 2),
    linear_op=px.constraints.identity,
)
block2 = px.BlockTerm(
    proximal=px.prox.L1(weight=0.2),
    linear_op=-jnp.eye(3),
)
problem = px.BlockProblem(blocks=(block1, block2), rhs=0.0)
x0 = (jnp.zeros(3), jnp.zeros(3))

sol = px.solve(
    problem,
    x0,
    solver=px.solvers.ADMM(opt_tol=1e-3, feas_tol=1e-3, rho=2.0),
)
print(sol.status_enum.name, float(sol.stats.obj_val))
```

---

## 2. Three-Block Nonconvex Quadratic Splitting

For nonconvex multi-block problems, `DPADMM` and `PAdaptADMM` provide
certified convergence guarantees using proximal inner subproblem solvers:

```python
import jax.numpy as jnp
from proxax import px

rhs = jnp.array([1.0, -0.5, 0.25])
a1 = jnp.eye(3)
a2 = jnp.array([[1.0, 0.5, 0.0], [0.0, 1.0, -0.5], [0.5, 0.0, 1.0]])
a3 = -jnp.eye(3)

b1 = px.BlockTerm(
    smooth=lambda x: (
        0.5 * jnp.sum(x**2) - 0.8 * jnp.sum(jnp.sin(x)) - jnp.sum(x)
    ),
    proximal=px.prox.Box(lower=-1.5, upper=1.5),
    linear_op=a1,
)
b2 = px.BlockTerm(
    smooth=lambda x: 0.5 * jnp.sum((x + 0.5) ** 2),
    proximal=px.prox.Box(lower=-1.5, upper=1.5),
    linear_op=a2,
)
b3 = px.BlockTerm(
    proximal=px.prox.Box(lower=-1.5, upper=1.5),
    linear_op=a3,
)
problem = px.BlockProblem(blocks=(b1, b2, b3), rhs=rhs)
x0 = (jnp.zeros(3), jnp.zeros(3), jnp.zeros(3))

sol = px.solve(
    problem,
    x0,
    solver=px.solvers.DPADMM(opt_tol=1e-3, feas_tol=1e-3, c_init=1.0),
)
print(sol.status_enum.name, float(sol.stats.obj_val))
```
