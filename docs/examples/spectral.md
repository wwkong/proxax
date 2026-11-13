# Low-Rank & Spectral Matrix Completion

This example recovers a noisy, partially observed low-rank matrix using
`SpectralProblem`, comparing nonconvex log-sum singular-value regularization
against a convex nuclear-norm baseline.

All code on this page can be executed via
`examples/spectral_matrix_completion.py`:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python examples/spectral_matrix_completion.py
```

---

## 1. Nonconvex Log-Sum Spectral Model

Given noisy partial observations $P_\Omega(M_{\text{true}} + E)$ of a low-rank
matrix $M_{\text{true}} \in \mathbb{R}^{p \times q}$, convex nuclear-norm
regularization $\bar{\mu}\|X\|_* = \bar{\mu}\|\sigma(X)\|_1$ shrinks all
singular values equally and biases large singular values downward. A nonconvex
log-sum spectral penalty avoids shrinking the dominant singular values while
still promoting exact low rank inside a Frobenius ball
$\mathcal{B}_R = \{X : \|X\|_F \le R\}$:

$$\min_{\|X\|_F \le R} \; \underbrace{\frac{1}{2}\|P_\Omega(X - M_{\text{obs}})\|_F^2}_{f_1(X)} + \alpha \beta \Big(1 - e^{-\|\sigma(X)\|_2^2 / (2\theta)}\Big) + \mu \sum_{i=1}^r \beta \log\!\left(1 + \frac{\sigma_i(X)}{\theta}\right)$$

Writing $\kappa_0 = \beta / \theta$ for the slope of $t \mapsto \beta \log(1 +
t/\theta)$ at $t = 0^+$, we split the singular-value penalty into a smooth
weakly convex vector term $f_2^{\mathcal{V}}$ and a convex nonsmooth vector
regularizer $h^{\mathcal{V}}$:

- $f_1(X) = \frac{1}{2}\|P_\Omega(X - M_{\text{obs}})\|_F^2$ with $(m_1, M_1) =
  (0, 1)$,
- $f_2^{\mathcal{V}}(s) = \alpha \beta (1 - e^{-\|s\|_2^2 / (2\theta)}) + \mu
  \sum_i \big[\beta \log(1 + |s_i|/\theta) - \kappa_0 |s_i|\big]$ (smooth on
  $\mathbb{R}^r$ because the kink of $|s_i|$ at $0$ cancels),
- $h^{\mathcal{V}}(s) = (\mu \kappa_0)\|s\|_1 + \delta_{\|\cdot\|_2 \le R}(s)$,
  whose 1D proximal operator is soft-thresholding followed by $\ell_2$-ball
  projection.

Because `DAICG` and `IAICG` exploit the `SpectralProblem` decomposition
(see {doc}`../foundations/problem_classes`), each outer iteration requires
**only one thin matrix SVD**, solving the inner proximal subproblem purely on
singular-value vectors in $\mathbb{R}^{\min(p, q)}$.

---

## 2. Implementation in `proxax`

We define the 1D singular-value proximal operator (`L1` + Euclidean ball) and
assemble `proxax.SpectralProblem`:

```python
import dataclasses
import jax
import jax.numpy as jnp
import optax
import proxax
from proxax import _typing
from proxax import prox
from proxax import solvers
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

Pass `spec_problem` to `IAICG`, `DAICG`, or any general unconstrained solver
such as `APD`:

```python
spec_problem = proxax.SpectralProblem(
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

For standard convex nuclear-norm matrix completion ($f_2^{\mathcal{V}} \equiv 0$),
`IAICG` and `DAICG` also accept a regular `CompositeProblem` with
`prox.NuclearNorm`:

```python
nuc_problem = proxax.CompositeProblem(
    smooth=terms.SmoothTerm(f1_fn, upper_curvature=1.0, lower_curvature=0.0),
    proximal=prox.NuclearNorm(weight=mu_reg * kappa0),
)
sol_nuc = proxax.solve(nuc_problem, x0, solver=solvers.DAICG(opt_tol=1e-3))
```

---

## 3. Numerical Results ($18 \times 15$, True Rank $r^\star = 3$, 65% Observed)

With Gaussian observation noise ($\sigma_{\text{noise}} = 0.05$) and stationarity
tolerance $\|\hat{V}\|_F \le 10^{-3}$, all nonconvex spectral solvers recover
the **exact rank $3$** of $M_{\text{true}}$ and cut full-matrix relative
reconstruction error $\|\hat{X} - M_{\text{true}}\|_F / \|M_{\text{true}}\|_F$
by more than **$3.5\times$** compared with convex nuclear-norm regularization:

| Model & Solver | Status | Objective | Certified $\|\hat{V}\|_F$ | Recovered Rank | Relative Error | Outer Iters |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Nonconvex `DAICG` (`monotone`) | `CONVERGED` | `4.6055` | `9.64e-04` | **`3 / 15`** | **`0.028`** | `117` |
| Nonconvex `IAICG` (`constant`) | `CONVERGED` | `4.6055` | `9.83e-04` | **`3 / 15`** | **`0.028`** | `152` |
| Nonconvex `IAICG` (`adaptive`) | `CONVERGED` | `4.6055` | `9.82e-04` | **`3 / 15`** | **`0.028`** | `197` |
| Nonconvex `APD` (matrix SVD) | `CONVERGED` | `4.6055` | `8.27e-04` | **`3 / 15`** | **`0.028`** | `10` |
| Convex `NuclearNorm` (`DAICG`) | `CONVERGED` | `20.5486` | `9.84e-04` | `4 / 15` | `0.101` | `164` |
