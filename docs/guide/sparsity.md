# Sparse Optimization with BCOO

Proxax supports native sparse linear algebra via JAX's `jax.experimental.sparse.BCOO` format without densifying constraint or Hessian matrices. This makes large-scale sparse QPs, linear programs, and batched sparse layers practical on both CPU and GPU/TPU.

## Sparse Constraints and Objectives

Pass `BCOO` matrices directly into your smooth objective and any `px.constraints` builder (`equality`, `less_equal`, `greater_equal`, `box`, `linear_cone`):

```python
import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import scipy.sparse as sp
import proxax as px

# Convert SciPy sparse matrices to deduplicated JAX BCOO
P_sp = sp.diags([4.0, 400.0, 0.04, 16.0], format="coo")
A_sp = sp.coo_matrix([[1.0, 100.0, 0.0, 0.0], [0.0, 0.0, 0.1, 4.0]])

P_bcoo = jsparse.BCOO.from_scipy_sparse(P_sp).sum_duplicates()
A_bcoo = jsparse.BCOO.from_scipy_sparse(A_sp).sum_duplicates()

q = jnp.array([-1.0, -50.0, -0.1, -4.0])
b = jnp.array([1.0, 1.0])
x0 = jnp.zeros(4)

prob = px.CompositeProblem(
    smooth=lambda x: 0.5 * jnp.dot(x, P_bcoo @ x) + jnp.dot(q, x),
    proximal=px.prox.NonNegative(),
    constraints=px.constraints.equality(A_bcoo, rhs=b),
)
```

Under the hood, `MatrixLinearOperator` detects `BCOO` and dispatches `A @ x` and `A.T @ y` using sparse matvecs (`BCOO.T @ y` for adjoints) without converting to dense arrays.

## Zero-Densification Modified Ruiz Equilibration

Ill-conditioned sparse problems benefit significantly from preconditioning. Pass the sparse Hessian `P_bcoo` to `px.ruiz_equilibrate` or `px.solve_preconditioned`:

```python
sol = px.solve_preconditioned(
    prob,
    x0,
    method="ialm",
    opt_tol=1e-6,
    feas_tol=1e-6,
    hessian=P_bcoo,
    time_limit=10.0,
)
```

`px.ruiz_equilibrate` equilibrates row and column norms of `A_bcoo` (and `P_bcoo`) using sparse reductions and diagonal scaling, keeping all stored operators in `BCOO` format (including each block of a `ProductConstraint`).

## Differentiable Sparse QP Layers

`px.layers.QPLayer` also accepts `BCOO` matrices for `Q` and `A`:

```python
layer = px.layers.QPLayer(
    Q=P_bcoo,
    A=A_bcoo,
    l=jnp.zeros(4),
    u=jnp.full(4, jnp.inf),
)
```

## Batched Sparse Solves

`px.batch_solve` and `px.batch_implicit_solve` treat `BCOO` leaves atomically when slicing parameter PyTrees across `backend="threads"` or `chunk_size > 1`, supporting batched `BCOO` arrays (`n_batch >= 1`):

```python
# params_batch has batched BCOO matrices with leading batch dimension B
sols = px.batch_solve(
    build_sparse_prob,
    x0,
    params_batch,
    method="ialm",
    precondition=True,
    backend="threads",
)
```

## Practical Tips

- **Call `.sum_duplicates()`**: Deduplicate and sort `BCOO` indices once before `jax.jit` compilation to avoid redundant scattered additions during matvecs.
- **Enable float64 for ill-conditioned QPs**: Set `JAX_ENABLE_X64=1` (or `jax.config.update("jax_enable_x64", True)`) before building `BCOO` arrays when solving tight-tolerance (`1e-6`) or poorly scaled sparse problems.
- **Combine with `time_limit`**: Large sparse solves can be bounded cleanly with `time_limit=...` in `px.solve` and `px.solve_preconditioned`.
