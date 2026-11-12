# Unconstrained Examples (`nc_opt`)

This tutorial ports the three core unconstrained composite benchmarks from
[`nc_opt/examples/unconstrained/`](https://github.com/wwkong/nc_opt/tree/master/examples/unconstrained)
and `nc_opt/docs/pages/examples.rst`. All code on this page can be executed
directly via `examples/unconstrained_nc_opt.py`:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python examples/unconstrained_nc_opt.py
```

---

## 1. Basic Convex Univariate QP (`basic_convex_qp.m`)

As a sanity check, consider the unconstrained convex quadratic program from
`examples/unconstrained/basic_convex_qp.m`:

$$\min_{x \in \mathbb{R}} \; \frac{1}{2} x^2 - x + \frac{1}{2}$$

whose unique minimizer is $x^\star = 1$ with optimal value $0$.

```python
import jax.numpy as jnp
import proxax
from proxax import solvers

problem = proxax.CompositeProblem(
    smooth=lambda x: 0.5 * jnp.sum(x**2) - jnp.sum(x) + 0.5,
)

sol = proxax.solve(problem, jnp.array([-3.0]), solver=solvers.APD(opt_tol=1e-6))
print(sol.status_enum.name, float(sol.x[0]), float(sol.stats.opt_residual))
# CONVERGED 1.0 0.0
```

Because `APD` performs an adaptive composite gradient refinement at step $0$, a
pure quadratic with isotropic Hessian $I$ is solved exactly during step-0
calibration with $\|\hat{v}\|_2 = 0$.

---

## 2. Nonconvex QP over the Unit Simplex (`nonconvex_qp.m`)

Next, consider the classic benchmark from Kong, Melo, and Monteiro (*SIAM J.
Optim.*, 2019) and `utils/generators/test_fn_unconstr_01.m`:

$$\min_{x \in \mathbb{R}^n} \; -\frac{\xi}{2}\|D B x\|_2^2 + \frac{\tau}{2}\|A x - b\|_2^2 + \delta_{\Delta^n}(x)$$

where $\Delta^n = \{x \in \mathbb{R}^n_+ : \sum_{i=1}^n x_i = 1\}$ is the
probability simplex and $(\tau, \xi) \in \mathbb{R}_{++}^2$ are calibrated so
the indefinite Hessian $H = \tau A^\top A - \xi B^\top D^\top D B$ attains a
prescribed upper smoothness $M = \lambda_{\max}(H)$ and weak-convexity constant
$m = -\lambda_{\min}(H) > 0$.

```python
import jax
import jax.numpy as jnp
import proxax
from proxax import prox
from proxax import solvers
from proxax.oracle import terms

key = jax.random.key(777)
k1, k2, k3, k4 = jax.random.split(key, 4)
dim_m, dim_n = 10, 20
d_diag = jax.random.randint(k1, (dim_n,), 1, 15).astype(jnp.float32)
a_mat = jax.random.uniform(k2, (dim_m, dim_n))
b_mat = jax.random.uniform(k3, (dim_n, dim_n))
b_vec = jax.random.uniform(k4, (dim_m,))
db_mat = d_diag[:, None] * b_mat

# Calibrate (tau, xi) so lambda_max(H) = 50.0 and -lambda_min(H) > 0
hp = a_mat.T @ a_mat
hn = db_mat.T @ db_mat
xi = 5.0 / float(jnp.max(jnp.linalg.eigvalsh(hn)))
tau = 50.0 / float(jnp.max(jnp.linalg.eigvalsh(hp)))
eigs = jnp.linalg.eigvalsh(0.5 * (tau * hp - xi * hn + (tau * hp - xi * hn).T))
m_upper = float(jnp.max(eigs))
m_lower = float(jnp.maximum(0.0, -jnp.min(eigs)))

def smooth_qp(x: jax.Array) -> jax.Array:
  res = a_mat @ x - b_vec
  dbx = db_mat @ x
  return 0.5 * tau * jnp.dot(res, res) - 0.5 * xi * jnp.dot(dbx, dbx)

problem = proxax.CompositeProblem(
    smooth=terms.SmoothTerm(
        smooth_qp, upper_curvature=m_upper, lower_curvature=m_lower
    ),
    proximal=prox.Simplex(scale=1.0),
)

x0 = jnp.ones(dim_n) / dim_n
for name, solver in [
    ("APD (param-free)", solvers.APD(opt_tol=1e-4)),
    ("AIPP", solvers.AIPP(opt_tol=1e-4)),
    ("ACACG", solvers.ACACG(opt_tol=1e-4)),
    ("AdapFISTA", solvers.AdapFISTA(opt_tol=1e-4)),
    ("NCFISTA", solvers.NCFISTA(opt_tol=1e-4)),
    ("ECG", solvers.ECG(opt_tol=1e-4)),
]:
  sol = proxax.solve(problem, x0, solver=solver)
  print(name, sol.status_enum.name, float(sol.stats.obj_val), int(sol.stats.grad_evals))
```

### Verified Solver Comparison (`opt_tol = 1e-4`, $M = 50.0$, $m = 0.34$)

All seven unconstrained composite solvers reach the same stationary point on
$\Delta^{20}$ with exact normal-cone subdifferential residual
$\|\hat{v}\|_2 \le 10^{-4}$, where $\hat{v} \in \nabla f(\hat{x}) + N_{\Delta^n}(\hat{x})$:

| Solver | Curvature Used | Status | Objective | Certified $\|\hat{v}\|_2$ | $\nabla f$ Evals |
| :--- | :--- | :--- | ---: | ---: | ---: |
| `ACACG` | Optional $M$ | `CONVERGED` | `0.29116` | `5.94e-05` | `46` |
| `ECG` | None | `CONVERGED` | `0.29116` | `9.13e-05` | `74` |
| `APD` (`PF.APD`) | **None** | `CONVERGED` | `0.29116` | `2.79e-05` | `156` |
| `AdapFISTA` | **None** | `CONVERGED` | `0.29116` | `9.87e-05` | `254` |
| `NCFISTA` | $(m, M)$ | `CONVERGED` | `0.29116` | `8.33e-05` | `640` |
| `AIPP` | $m$ | `CONVERGED` | `0.29116` | `8.17e-05` | `927` |
| `UPFAG` | **None** | `CONVERGED` | `0.29116` | `3.99e-05` | `1751` |

---

## 3. Nonconvex Hyperbolic-Tangent SVM (`nonconvex_svm.m`)

From `examples/unconstrained/nonconvex_svm.m` and
`utils/generators/test_fn_svm_01.m`, consider binary classification with the
smooth nonconvex sigmoid/hyperbolic-tangent margin loss:

$$\min_{x \in \mathbb{R}^n} \; \frac{1}{k}\sum_{i=1}^k \Big[1 - \tanh\big(v_i \langle u_i, x\rangle\big)\Big] + \frac{1}{2k}\|x\|_2^2$$

where $(u_i, v_i) \in \mathbb{R}^n \times \{-1, +1\}$. Because $t \mapsto 1 -
\tanh(t)$ has second derivative $2 \tanh(t)\operatorname{sech}^2(t) \in
[-4\sqrt{3}/9, +4\sqrt{3}/9]$, the objective is nonconvex with worst-case bound
$M = \frac{4\sqrt{3}}{9 k}\|U\|_F^2 + \frac{1}{k}$, yet its local curvature along
the trajectory is much smaller.

```python
import jax
import jax.numpy as jnp
import proxax
from proxax import solvers

k1, k2 = jax.random.split(jax.random.key(42))
dim_n, num_samples = 25, 60
u_mat = jax.random.normal(k1, (dim_n, num_samples)) / jnp.sqrt(dim_n)
x_true = jax.random.normal(k2, (dim_n,))
labels = jnp.where(u_mat.T @ x_true >= 0.0, 1.0, -1.0)
reg = 1.0 / num_samples

def svm_loss(x: jax.Array) -> jax.Array:
  margins = labels * (u_mat.T @ x)
  return jnp.mean(1.0 - jnp.tanh(margins)) + 0.5 * reg * jnp.dot(x, x)

# Parameter-free solvers require zero curvature bounds:
problem = proxax.CompositeProblem(smooth=svm_loss)
sol = proxax.solve(problem, jnp.zeros(dim_n), solver=solvers.APD(opt_tol=1e-4))
```

### Verified Solver Comparison (`opt_tol = 1e-4`)

| Solver | Curvature Supplied? | Status | Loss | $\|\nabla f(\hat{x})\|_2$ | Train Acc | $\nabla f$ Evals |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| `AdapAPG` | No | `CONVERGED` | `0.60651` | `8.70e-05` | `95.0%` | `35` |
| `APD` (`PF.APD`) | No | `CONVERGED` | `0.60651` | `8.60e-05` | `95.0%` | `60` |
| `AG` | Yes ($M$) | `CONVERGED` | `0.60651` | `9.87e-05` | `95.0%` | `364` |
| `UPFAG` | No | `CONVERGED` | `0.60651` | `9.90e-05` | `95.0%` | `380` |
