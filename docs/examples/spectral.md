# Spectral Matrix Completion (`nc_opt`)

This tutorial ports the nonconvex spectral matrix completion benchmark from
[`nc_opt/examples/spectral/nonconvex_spectral_mc.m`](https://github.com/wwkong/nc_opt/blob/master/examples/spectral/nonconvex_spectral_mc.m)
and `utils/generators/test_fn_spectral_mc_01.m` (Kong & Monteiro,
*arXiv:2007.11772*, 2020), and compares it directly against a convex nuclear-norm
baseline. All code on this page can be executed via
`examples/spectral_matrix_completion.py`:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python examples/spectral_matrix_completion.py
```

---

## 1. Nonconvex Log-Sum Spectral Matrix Completion Model

Given noisy partial observations $P_\Omega(M_{\text{true}} + E)$ of a low-rank
matrix $M_{\text{true}} \in \mathbb{R}^{m \times n}$, convex nuclear-norm
regularization $\bar{\mu}\|X\|_* = \bar{\mu}\|\sigma(X)\|_1$ shrinks all
singular values equally and biases the leading singular values downward. Following
Kong and Monteiro (2020), we instead minimize the nonconvex spectral objective
over the Frobenius ball $\mathcal{B}_R = \{X : \|X\|_F \le R\}$:

$$\min_{\|X\|_F \le R} \; \underbrace{\frac{1}{2}\|P_\Omega(X - M_{\text{obs}})\|_F^2}_{f_1(X)} + \alpha \beta \Big(1 - e^{-\|\sigma(X)\|_2^2 / (2\theta)}\Big) + \mu \sum_{i=1}^r \beta \log\!\left(1 + \frac{\sigma_i(X)}{\theta}\right)$$

Using $\kappa_0 = \beta / \theta$ (the slope of $t \mapsto \beta \log(1 +
t/\theta)$ at $t = 0^+$), we split the singular-value penalty into a smooth
weakly-convex vector term $f_2^V$ and a convex nonsmooth vector regularizer
$h^V$:

- $f_1(X) = \frac{1}{2}\|P_\Omega(X - M_{\text{obs}})\|_F^2$ with $(m_1, M_1) =
  (0, 1)$,
- $f_2^V(s) = \alpha \beta (1 - e^{-\|s\|_2^2 / (2\theta)}) + \mu \sum_i
  \big[\beta \log(1 + |s_i|/\theta) - \kappa_0 |s_i|\big]$, which is smooth on
  $\mathbb{R}^r$ because the kink of $|s_i|$ at $0$ cancels out,
- $h^V(s) = (\mu \kappa_0)\|s\|_1 + \delta_{\|\cdot\|_2 \le R}(s)$, whose 1D
  proximal operator is soft-thresholding followed by Euclidean ball projection.

### Why `SpectralProblem` + `IAICG` / `DAICG`?

A general matrix solver (such as `APD` on `CompositeProblem`) must compute a
full matrix SVD on **every inner line-search trial**. By contrast, `IAICG` and
`DAICG` factorize $Z_0 - \lambda \nabla f_1^\xi(Z_0) = P \operatorname{diag}(x_0)
Q^\top$ with **only one SVD per outer iteration**, solve the resulting 1D
singular-value subproblem in $\mathbb{R}^{\min(m, n)}$ via `ACG`, and construct
an exact matrix subgradient $\hat{V} \in \nabla f_1(\hat{Z}) + \nabla
(f_2^V \circ \sigma)(\hat{Z}) + \partial (h^V \circ \sigma)(\hat{Z})$ via
`refine_icg`.

---

## 2. Implementation in `proxax`

```python
import dataclasses
import jax
import jax.numpy as jnp
import optax
import proxax
from proxax import _typing
from proxax import oracle
from proxax import prox
from proxax import solvers
from proxax import tree
from proxax.oracle import terms

@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class BallL1VectorProx:
  """Singular-value prox for weight * ||s||_1 + delta_{||s||_2 <= radius}."""

  weight: float
  radius: float

  @property
  def strong_convexity(self) -> float:
    return 0.0

  def value(self, x: _typing.PyTree) -> jax.Array:
    return self.weight * jnp.sum(jnp.abs(jnp.asarray(x)))

  def prox(self, x: _typing.PyTree, step_size: _typing.Scalar) -> _typing.PyTree:
    s_vec = jnp.asarray(x)
    shrunk = jnp.sign(s_vec) * jnp.maximum(
        jnp.abs(s_vec) - step_size * self.weight, 0.0
    )
    return optax.projections.projection_l2_ball(shrunk, self.radius)

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    return optax.projections.projection_l2_ball(jnp.asarray(x), self.radius)
```

We assemble `oracle.SpectralProblem` and pass it directly to `IAICG`, `DAICG`,
or any general unconstrained `Solver` (such as `APD`):

```python
spec_problem = oracle.SpectralProblem(
    f1=terms.SmoothTerm(f1_fn, upper_curvature=1.0, lower_curvature=0.0),
    f2_vec=terms.SmoothTerm(
        f2_vec_fn, upper_curvature=m2_upper, lower_curvature=m2_lower
    ),
    prox_vec=BallL1VectorProx(weight=mu_reg * kappa0, radius=radius),
    domain_proj=ball_proj,
)

sol_ia = proxax.solve(
    spec_problem, x0, solver=solvers.IAICG(opt_tol=1e-3, step_type="adaptive")
)
sol_da = proxax.solve(
    spec_problem, x0, solver=solvers.DAICG(opt_tol=1e-3, monotone=True)
)
```

For convex nuclear-norm problems without a nonconvex $f_2^V$ term, `IAICG` and
`DAICG` also accept a standard `CompositeProblem` with `prox.NuclearNorm`:

```python
nuc_problem = proxax.CompositeProblem(
    smooth=terms.SmoothTerm(f1_fn, upper_curvature=1.0, lower_curvature=0.0),
    proximal=prox.NuclearNorm(weight=mu_reg * kappa0),
)
sol_nuc = proxax.solve(nuc_problem, x0, solver=solvers.DAICG(opt_tol=1e-3))
```

---

## 3. Verified Numerical Comparison ($18 \times 15$, True Rank $r^\star = 3$, 65% Observed)

With additive Gaussian noise ($\sigma_{\text{noise}} = 0.05$) and tolerance
$\|\hat{V}\|_F \le 10^{-3}$, all nonconvex solvers recover the **exact rank
$3$** of $M_{\text{true}}$ and reduce full-matrix relative reconstruction error
$\|\hat{X} - M_{\text{true}}\|_F / \|M_{\text{true}}\|_F$ by more than **$3.5\times$**
compared with convex nuclear-norm regularization:

| Model & Solver | Status | Objective | Certified $\|\hat{V}\|_F$ | Recovered Rank | Relative Error | Outer Iters |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Nonconvex `DAICG` (`monotone`) | `CONVERGED` | `4.6055` | `9.64e-04` | **`3 / 15`** | **`0.028`** | `117` |
| Nonconvex `IAICG` (`constant`) | `CONVERGED` | `4.6055` | `9.83e-04` | **`3 / 15`** | **`0.028`** | `152` |
| Nonconvex `IAICG` (`adaptive`) | `CONVERGED` | `4.6055` | `9.82e-04` | **`3 / 15`** | **`0.028`** | `197` |
| Nonconvex `APD` (matrix SVD) | `CONVERGED` | `4.6055` | `8.27e-04` | **`3 / 15`** | **`0.028`** | `10` |
| Convex `NuclearNorm` (`DAICG`) | `CONVERGED` | `20.5486` | `9.84e-04` | `4 / 15` | `0.101` | `164` |
