# Constrained QP, Sparse PCA & Matrix Completion

This page demonstrates set- and cone-constrained optimization in Proxax on
three problems:

1. **Linearly-constrained nonconvex QP** over the probability simplex, comparing
   all five constrained solvers (`IALM`, `IAIPAL`, `QPAPD`, `SProxALM`,
   `QPAIPP`).
2. **Consensus Sparse PCA on the $k$-Fantope**, combining a custom matrix
   projector, `px.prox.MultiProx`, and a matrix-free dictionary PyTree
   constraint.
3. **Box-constrained nonconvex matrix completion** with entrywise rating bounds.

All code on this page can be executed directly via
`examples/constrained.py`:

```bash
uv run python examples/constrained.py
```

---

## 1. Linearly-Constrained Nonconvex Simplex QP

We minimize an indefinite quadratic over the intersection of the probability
simplex $\Delta^n$ and an affine subspace $A_{\text{eq}} x = b$:

$$\min_{x \in \mathbb{R}^n} \; \frac{\tau}{2}\|C x - d\|_2^2 - \frac{\xi}{2}\|D B x\|_2^2 + \delta_{\Delta^n}(x) \quad \text{s.t.} \quad A_{\text{eq}} x = b$$

Every constrained solver returns $(\hat{x}, \hat{p}, \hat{v})$ satisfying exact
Lagrangian subdifferential membership:

$$\hat{v} \in \nabla f(\hat{x}) + N_{\Delta^n}(\hat{x}) + A_{\text{eq}}^\top \hat{p}, \qquad \|\hat{v}\|_2 \le \rho, \qquad \|A_{\text{eq}} \hat{x} - b\|_2 \le \eta.$$

```python
import jax
import jax.numpy as jnp
from proxax import px

key = jax.random.key(777)
k1, k2, k3, k4, k5 = jax.random.split(key, 5)
dim_eq, dim_n = 4, 12
d_diag = jax.random.randint(k1, (dim_n,), 1, 10).astype(jnp.float32)
c_mat = jax.random.uniform(k2, (dim_eq, dim_n))
b_mat = jax.random.uniform(k3, (dim_n, dim_n))
a_eq = jax.random.uniform(k4, (dim_eq, dim_n))
d_vec = jax.random.uniform(k5, (dim_eq,))

db_mat = d_diag[:, None] * b_mat
hp, hn = c_mat.T @ c_mat, db_mat.T @ db_mat
xi = 2.0 / float(jnp.max(jnp.linalg.eigvalsh(hn)))
tau = 20.0 / float(jnp.max(jnp.linalg.eigvalsh(hp)))
eigs = jnp.linalg.eigvalsh(0.5 * (tau * hp - xi * hn + (tau * hp - xi * hn).T))
m_upper = float(jnp.maximum(0.1, jnp.max(eigs)))
m_lower = float(jnp.maximum(0.1, -jnp.min(eigs)))
rhs_b = a_eq @ (jnp.ones(dim_n) / dim_n)

def smooth_fn(x: jax.Array) -> jax.Array:
  res = c_mat @ x - d_vec
  dbx = db_mat @ x
  return 0.5 * tau * jnp.dot(res, res) - 0.5 * xi * jnp.dot(dbx, dbx)

problem = px.CompositeProblem(
    smooth=px.oracle.SmoothTerm(
        smooth_fn, upper_curvature=m_upper, lower_curvature=m_lower
    ),
    proximal=px.prox.Simplex(scale=1.0),
    constraints=px.constraints.equality(a_eq, rhs=rhs_b),
)
```

### Solver Comparison (`opt_tol = feas_tol = 5e-4`)

| Solver | Method Family | Status | Objective | $\|\hat{v}\|_2$ | $\|A_{\text{eq}}\hat{x} - b\|_2$ | $\nabla f$ Evals |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| `IALM` | Parameter-free inexact ALM | `CONVERGED` | `-0.04166` | `3.96e-04` | `3.62e-04` | `196` |
| `IAIPAL` | Single-step proximal ALM (`m` + `M`) | `CONVERGED` | `-0.04145` | `2.97e-04` | `3.62e-06` | `251` |
| `QPAPD` | Parameter-free quadratic penalty | `CONVERGED` | `-0.04164` | `4.70e-04` | `3.04e-04` | `796` |
| `SProxALM` | Single-loop smoothed proximal ALM | `CONVERGED` | `-0.04139` | `4.90e-04` | `1.10e-04` | `1016` |
| `QPAIPP` | Quadratic penalty + adaptive `AIPP` | `CONVERGED` | `-0.04164` | `3.83e-04` | `3.04e-04` | `2719` |

---

## 2. Consensus Sparse PCA on the $k$-Fantope

Sparse Principal Component Analysis seeks a rank-$k$ projection matrix with
sparse entries using the nonconvex Minimax Concave Penalty (MCP) over the
$k$-Fantope $\mathcal{F}^k = \{\Pi \in \mathbb{S}_+^p : 0 \preceq \Pi \preceq I,\,
\operatorname{tr}(\Pi) = k\}$. Splitting the MCP penalty into its convex
$\ell_1$ part $\nu \|\Phi\|_1$ and smooth $1/b$-weakly convex concave part
$q_{b,\nu}(\Phi)$ yields a two-block consensus problem:

$$\min_{\Pi, \Phi \in \mathbb{R}^{p \times p}} \; -\langle \widehat{\Sigma}, \Pi \rangle + q_{b,\nu}(\Phi) + \delta_{\mathcal{F}^k}(\Pi) + \nu \|\Phi\|_1 \quad \text{s.t.} \quad \Pi - \Phi = 0$$

where $q_{b,\nu}(t) = -t^2 / (2b)$ if $|t| \le b \nu$ and
$\frac{1}{2} b \nu^2 - \nu |t|$ otherwise.

In Proxax, we keep the variable as a dictionary PyTree `{"pi": ..., "phi": ...}`,
pair the two block regularizers via `px.prox.MultiProx`, and write the matrix
consensus constraint as a simple linear function `z["pi"] - z["phi"]`:

```python
import dataclasses
import typing
import jax
import jax.numpy as jnp
from proxax import px

@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class FantopeProx:
  """Indicator and Euclidean projector of the k-Fantope."""

  rank_k: float

  @property
  def strong_convexity(self) -> float:
    return 0.0

  def value(self, x: px.typing.PyTree) -> jax.Array:
    return jnp.asarray(0.0, dtype=jnp.asarray(x).dtype)

  def _project(self, x: px.typing.PyTree) -> jax.Array:
    mat = jnp.asarray(x)
    eigvals, eigvecs = jnp.linalg.eigh(0.5 * (mat + mat.T))
    lo, hi = jnp.min(eigvals) - 1.0, jnp.max(eigvals)

    def body_fn(
        _: int, bounds: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
      low, high = bounds
      mid = 0.5 * (low + high)
      s = jnp.sum(jnp.clip(eigvals - mid, 0.0, 1.0))
      res = jax.lax.cond(s > self.rank_k, lambda: (mid, high), lambda: (low, mid))
      return typing.cast(tuple[jax.Array, jax.Array], res)

    lo_f, hi_f = jax.lax.fori_loop(0, 36, body_fn, (lo, hi))
    proj_eigs = jnp.clip(eigvals - 0.5 * (lo_f + hi_f), 0.0, 1.0)
    return typing.cast(jax.Array, (eigvecs * proj_eigs[None, :]) @ eigvecs.T)

  def prox(self, x: px.typing.PyTree, step_size: px.typing.Scalar) -> px.typing.PyTree:
    del step_size
    return self._project(x)

  def domain_project(self, x: px.typing.PyTree) -> px.typing.PyTree:
    return self._project(x)
```

With ground-truth rank-1 spike $v^\star = \frac{1}{\sqrt{3}}(1, 1, 1, 0, 0, 0,
0, 0)^\top$ and covariance $\widehat{\Sigma} = 8 v^\star (v^\star)^\top + 0.25
I_8$, `IALM` recovers the exact $3 \times 3$ support (`9/64` nonzeros) and unit
subspace cosine similarity:

```text
IALM -> status=CONVERGED, stat=8.49e-04, ||Pi - Phi||_F=5.29e-04, subspace_cos=1.0000, nnz(Phi)=9/64
```

---

## 3. Box-Constrained Nonconvex Matrix Completion

When matrix entries represent bounded ratings in $[0, 2]$ under an entrywise
smooth nonconvex log-penalty:

$$\min_{X \in \mathbb{R}^{m \times n}} \; \frac{1}{2}\|P_\Omega(X - M_{\text{obs}})\|_F^2 + \mu \beta \sum_{i,j} \log\!\left(1 + \frac{\sqrt{X_{ij}^2 + \epsilon^2}}{\theta}\right) \quad \text{s.t.} \quad 0 \le X_{ij} \le 2$$

we can enforce the box constraint via `px.constraints.box` and solve
parameter-free with `QPAPD`:

```python
problem = px.CompositeProblem(
    smooth=smooth_bmc,
    constraints=px.constraints.box(
        px.constraints.identity, lower=0.0, upper=2.0
    ),
)
sol = px.solve(
    problem,
    jnp.full((12, 10), 0.5),
    solver=px.solvers.QPAPD(opt_tol=1e-3, feas_tol=1e-4),
)
# QPAPD -> status=CONVERGED, stat=2.48e-04, box_viol=3.76e-05, full_matrix_RMSE=0.4686
```
