# Getting Started & Core Concepts

`proxax` provides **certified first-order proximal point methods** for
unconstrained and set/cone-constrained nonconvex composite optimization in
**JAX** and **Optax**, porting the complete suite of algorithms from
[`wwkong/nc_opt`](https://github.com/wwkong/nc_opt).

---

## Installation

::::{tab-set}

:::{tab-item} pip (CPU)
```bash
pip install proxax
```
:::

:::{tab-item} uv
```bash
uv add proxax
```
:::

:::{tab-item} NVIDIA GPU (CUDA 12 / 13)
```bash
# CUDA 13
pip install "proxax[cuda13]"

# CUDA 12
pip install "proxax[cuda12]"
```
:::

::::

---

## Why Certified Stationarity?

Many first-order optimization packages stop on heuristic step-length or
raw gradient norms that do not apply when a nonsmooth regularizer $h$ or
indicator constraint $\delta_C$ is present.

Every solver in `proxax` returns a **`Solution`** containing a **certified
refined primal-dual pair** $(\hat{x}, \hat{p}, \hat{v})$ such that:

1. **Exact Domain Membership:** $\hat{x} \in \operatorname{dom}(h)$ (e.g.,
   strictly on the unit simplex $\Delta^n$, box $[l, u]$, or $k$-Fantope).
2. **Exact Subdifferential Inclusion:**

   $$\hat{v} \in \nabla f(\hat{x}) + \partial h(\hat{x}) + \mathcal{A}^*(\hat{p})$$

   computed analytically via a single composite gradient refinement step (or
   zero-overhead inner certificate in `APD`, `IAICG`, and `DAICG`).
3. **Certified Termination:** `SolveStatus.CONVERGED` is reported **only** when
   both $\|\hat{v}\| \le \rho_{\text{opt}}$ and
   $\|\mathcal{A}(\hat{x}) - \Pi_{\mathcal{C}}(\mathcal{A}(\hat{x}))\| \le \eta_{\text{feas}}$.

---

## Anatomy of a `CompositeProblem`

A `CompositeProblem` models:

$$\min_{x \in \mathcal{X}} \; f(x) + h(x) \quad \text{s.t.} \quad \mathcal{A}(x) \in \mathcal{C}$$

where the variable $x \in \mathcal{X}$ can be **any JAX PyTree** (`jax.Array`,
nested `dict`, `tuple`, or registered `@dataclass`):

```python
import jax.numpy as jnp
import proxax
from proxax import constraints
from proxax import prox
from proxax import solvers

# 1. Define smooth objective f(x) (differentiated automatically via jax.grad)
q_mat = jnp.array([[3.0, -1.5], [-1.5, -1.0]])  # Indefinite (nonconvex!)
c_vec = jnp.array([-0.8, 0.4])

problem = proxax.CompositeProblem(
    smooth=lambda x: 0.5 * jnp.dot(x, q_mat @ x) + jnp.dot(c_vec, x),
    proximal=prox.Box(lower=-1.0, upper=1.0),
    constraints=constraints.equality(lambda x: jnp.sum(x), rhs=jnp.asarray(0.5)),
)

# 2. Solve with parameter-free QP-APD (no curvature constants required!)
x0 = jnp.zeros(2)
sol = proxax.solve(problem, x0, solver=solvers.QPAPD(opt_tol=1e-5, feas_tol=1e-5))

print("Status:", sol.status_enum.name)
print("Certified x*:", sol.x)
print("Stationarity ||v_hat||_2:", float(sol.stats.opt_residual))
print("Feasibility  ||Ax - b||_2:", float(sol.stats.feas_residual))
```

### Supplying Optional Curvature Bounds `(m, M)`

If $f$ has known upper smoothness $M \ge 0$ and weak-convexity constant $m \ge 0$
(meaning $f(x) + \frac{m}{2}\|x\|_2^2$ is convex and $\nabla f$ is
$\max(m, M)$-Lipschitz), wrap $f$ in `proxax.oracle.terms.SmoothTerm`:

```python
from proxax.oracle import terms

smooth = terms.SmoothTerm(
    fn=lambda x: 0.5 * jnp.dot(x, q_mat @ x) + jnp.dot(c_vec, x),
    upper_curvature=3.6,
    lower_curvature=1.6,
)
```

Parameter-free solvers (`APD`, `QPAPD`, `IALM`, `UPFAG`, `AdapFISTA`, `AdapAPG`)
estimate curvatures automatically when omitted.

---

## Choosing a Solver

### Unconstrained Composite (`CompositeProblem`, `constraints=None`)

| Solver | Curvature Needed? | Best For | Reference |
| :--- | :--- | :--- | :--- |
| `APD` (`PF.APD`) | **None** (parameter-free) | Default general-purpose nonconvex & convex composite problems; optimal complexity in both regimes | Kong (SIAM J. Optim. 2024) |
| `AIPP` (`R-AIPP`) | Optional `m` (adaptive $\lambda_k$) | Nonconvex composite problems with known or estimated weak convexity $m$ | Kong, Melo & Monteiro (2019, 2020) |
| `ACACG` | Optional `M` | Fast average-curvature acceleration when local smoothness is much smaller than worst-case $M$ | Liang & Monteiro (2021) |
| `AdapFISTA` | **None** | Parameter-free FISTA-style acceleration on nonconvex composite objectives | Liang, Monteiro & Sim (2022) |
| `NCFISTA` | `(m, M)` or `M` | Nonconvex FISTA with known curvature bounds | Liang, Monteiro & Sim (2022) |
| `AdapAPG` | **None** (optional $\mu$) | Adaptive Nesterov APG with backtracking/relaxation | Lin & Xiao (2014) |
| `UPFAG` | **None** | Unified parameter-free Ghadimi–Lan accelerated gradient | Lan, Ouyang & Zhang (2023) |
| `AG` | Optional `M` | Classic Ghadimi–Lan nonconvex accelerated gradient | Ghadimi & Lan (2016) |
| `ECG` | **None** | Simple baseline adaptive proximal gradient / ISTA | Composite backtracking |

### Unconstrained Matrix Spectral (`SpectralProblem` / `NuclearNorm`)

| Solver | Best For | Key Mechanism | Reference |
| :--- | :--- | :--- | :--- |
| `DAICG` | Large-scale low-rank / log-sum / SCAD / MCP matrix completion | Doubly accelerated: outer FISTA extrapolation + 1D singular-value inner `ACG` (1 SVD per outer step!) | Kong & Monteiro (2020) |
| `IAICG` | Nonconvex spectral problems with constant or adaptive $\lambda_k$ | Reduces matrix prox subproblems to $O(\min(m, n))$ vector `ACG` steps | Kong & Monteiro (2020) |

### Set / Cone-Constrained (`CompositeProblem` with `constraints=...`)

| Solver | Curvature Needed? | Dual Multipliers? | Best For | Reference |
| :--- | :--- | :--- | :--- | :--- |
| `IALM` (`NL-IAPIAL` / `AIDAL`) | **None** (with `APD` inner) | Yes (`dampen` $\theta \in [0, 1)$) | General equality, box, inequality, and cone constraints with moderate penalty growth | Kong, Melo & Monteiro (2023) |
| `QPAPD` | **None** | Effective $\hat{p}_k = c_k(w - \Pi_{\mathcal{C}}(w))$ | Completely parameter-free quadratic penalty method; optimal $\tilde{\mathcal{O}}(\varepsilon^{-3})$ / $\tilde{\mathcal{O}}(\varepsilon^{-1.5})$ | Kong (SIAM J. Optim. 2024) |
| `QPAIPP` | Optional `m` | Effective $\hat{p}_k$ | Linearly/cone-constrained nonconvex QPs with `AIPP` subproblems | Kong, Melo & Monteiro (2019, 2020) |
| `IAIPAL` | Optional `(m, M)` | Yes (classical $\theta = 0$) | Single-`ACG`-step-per-outer-iteration proximal AL | Kong, Melo & Monteiro (SIAM J. Optim. 2023) |
| `SProxALM` | Optional `(L, \|A\|)` | Yes | Single-loop smoothed proximal ALM for polyhedral/box/equality constraints | Zhang & Luo (SIAM J. Optim. 2020) |
