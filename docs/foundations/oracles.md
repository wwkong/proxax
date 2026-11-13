# Oracles & Stationarity Certificates

In first-order optimization, an **oracle** is the mathematical and programmatic
interface through which a solver queries an optimization problem at trial points
$x \in \mathcal{X}$. Rather than treating $\phi(x) = f(x) + h(x)$ as an opaque
black box, Proxax uses **structured composite oracles** that expose the smooth
gradient $\nabla f$, the nonsmooth proximal mapping $\operatorname{prox}_{\tau h}$,
and the constraint Moreau envelope separately.

This page explains how every oracle in `proxax.oracle` works mathematically, how
solvers transform one oracle into another without re-tracing JAX functions under
`jax.jit`, and how `proxax.certificates` computes exact subgradient certificates.

---

## 1. Smooth Oracles (`SmoothObjective` & `SmoothTerm`)

A **smooth oracle** represents a differentiable function
$f : \mathcal{X} \to \mathbb{R}$ and provides:

- **Zero-order query:** $x \mapsto f(x) \in \mathbb{R}$,
- **First-order query:** $x \mapsto \big(f(x), \nabla f(x)\big) \in \mathbb{R} \times \mathcal{X}$,
- **Optional curvature metadata:** upper smoothness $M \ge 0$
  (`curvature_upper`) and weak convexity $m \ge 0$ (`curvature_lower`).

In `px.oracle`, any Python callable `fn(x)` passed to
`px.CompositeProblem(smooth=fn)` is automatically wrapped in `SmoothTerm`, which
computes `(f(x), grad_f(x))` via `jax.value_and_grad(fn)`. You can also
construct `px.oracle.SmoothTerm` directly when you want to supply known
curvatures or a hand-optimized `(value, grad)` kernel:

```python
import jax
import jax.numpy as jnp
from proxax import px

a_mat = jnp.array([[2.0, -0.5], [-0.5, -1.0]])


def quad_fn(x: jax.Array) -> jax.Array:
  return 0.5 * jnp.dot(x, a_mat @ x)


# Option A: Let JAX differentiate `quad_fn` automatically, with optional curvatures
smooth_auto = px.oracle.SmoothTerm(
    quad_fn,
    upper_curvature=2.1,
    lower_curvature=1.1,
)

# Option B: Supply a custom value-and-gradient function (`vag_fn`) to reuse `a_mat @ x`
def custom_vag(x: jax.Array) -> tuple[jax.Array, jax.Array]:
  ax = a_mat @ x
  return 0.5 * jnp.dot(x, ax), ax


smooth_custom = px.oracle.SmoothTerm(
    quad_fn,
    vag_fn=custom_vag,
)
```

When $f \equiv 0$ (pure feasibility or proximal problems), `smooth=None` uses
`px.oracle.ZeroSmooth()`, which has exact $(m, M) = (0, 0)$ and avoids autodiff.

---

## 2. Proximal Oracles (`ProxOperator`)

A **proximal oracle** represents a proper, closed, $\mu$-strongly convex
function $h : \mathcal{X} \to (-\infty, +\infty]$ and implements four operations
defined by `proxax.prox.ProxOperator`:

| Member | Mathematical Definition | Role in Solvers |
| :--- | :--- | :--- |
| `strong_convexity` | $\mu \ge 0$ | Accelerates inner `ACG` / `FISTA` steps when $\mu > 0$ (e.g., `L2Squared`, `ElasticNet`). |
| `value(x)` | $h(x)$ for $x \in \operatorname{dom} h$ | Evaluates composite descent $\phi(x) = f(x) + h(x)$. Indicator sets return `0.0` on feasible points. |
| `prox(x, step_size)` | $\operatorname{argmin}_{u} \big\{ h(u) + \frac{1}{2\tau}\|u - x\|_2^2 \big\}$ ($\tau = \text{step\_size}$) | Computes the exact resolvent $(I + \tau \partial h)^{-1}(x)$. |
| `domain_project(x)` | $\Pi_{\operatorname{cl}(\operatorname{dom} h)}(x) = \operatorname{argmin}_{u \in \operatorname{cl}(\operatorname{dom} h)} \|u - x\|_2^2$ | Projects the user's initial guess `x0` (or extrapolated points) onto $\operatorname{dom} h$ so $f$ and $h$ are never evaluated outside their domain. |

:::{important}
Because `value(x)` is only ever called at points produced by `prox(...)` or
`domain_project(...)`, indicator operators such as `prox.Box`, `prox.Simplex`,
and `prox.L2Ball` simply return `0.0` from `value(x)` — avoiding fragile
floating-point `if x in set else +inf` branches inside `jax.jit`.
:::

### Writing a Custom `ProxOperator`

Any immutable class registered as a JAX PyTree with those four members works
seamlessly across every solver and Optax transform:

```python
import dataclasses
import jax
import jax.numpy as jnp
from proxax import px

@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class WeightedL1:
  """Elementwise weighted L1 regularizer sum_i w_i |x_i|."""

  weights: jax.Array

  @property
  def strong_convexity(self) -> float:
    return 0.0

  def value(self, x: px.typing.PyTree) -> jax.Array:
    return jnp.sum(self.weights * jnp.abs(jnp.asarray(x)))

  def prox(self, x: px.typing.PyTree, step_size: px.typing.Scalar) -> px.typing.PyTree:
    u = jnp.asarray(x)
    thresh = step_size * self.weights
    return jnp.sign(u) * jnp.maximum(jnp.abs(u) - thresh, 0.0)

  def domain_project(self, x: px.typing.PyTree) -> px.typing.PyTree:
    return x
```

---

## 3. Composite Oracles & Subproblem Transformations

`CompositeOracle(smooth=f, proximal=h)` pairs a smooth oracle and a proximal
oracle to represent $\phi(x) = f(x) + h(x)$ (`CompositeOracleProtocol`).

Proximal-point, penalty, and augmented Lagrangian algorithms work by solving a
sequence of **modified composite subproblems** whose centers $z_k$, weights
$\lambda_k$, penalties $c_k$, and dual multipliers $p_k$ change at every outer
iteration. If a solver constructed Python closures `lambda x: f(x) + ...` inside
a loop, JAX would either fail inside `jax.lax.while_loop` or re-trace XLA
kernels repeatedly. Instead, `proxax.oracle` provides **PyTree-parameterized
oracle wrappers** that transform `CompositeOracleProtocol` instances algebraically:

### `ProxRegularizedOracle` (Proximal-Point Subproblems in `AIPP` & `IALM`)

Given proximal center $z_k \in \mathcal{X}$ and regularization split
$\frac{1}{\lambda_k} = \alpha_{\text{smooth}} + \alpha_{\text{prox}}$,
`ProxRegularizedOracle` forms the strongly convex composite subproblem:

$$f_{\text{sub}}(x) := f(x) + \frac{\alpha_{\text{smooth}}}{2}\|x - z_k\|_2^2, \qquad h_{\text{sub}}(x) := h(x) + \frac{\alpha_{\text{prox}}}{2}\|x - z_k\|_2^2.$$

- When $f$ has known weak convexity $m$, setting $\alpha_{\text{smooth}} = m$
  makes $f_{\text{sub}}$ convex, and $h_{\text{sub}}$ has guaranteed strong
  convexity $\mu + \alpha_{\text{prox}}$.
- Its proximal operator has an exact closed form in terms of $\operatorname{prox}_h$:

  $$\operatorname{prox}_{\tau h_{\text{sub}}}(u) = \operatorname{prox}_{\frac{\tau}{1 + \tau \alpha_{\text{prox}}} h}\!\left(\frac{u + \tau \alpha_{\text{prox}} z_k}{1 + \tau \alpha_{\text{prox}}}\right).$$

### `RescaledProxOracle` (Parameter-Free Subproblems in `APD`)

Given an adaptive lower-curvature estimate $\hat{m}_k > 0$ and scale
$s_k = \frac{1}{2\hat{m}_k}$, `RescaledProxOracle` forms the unit-regularized
subproblem:

$$\psi^s(z) := s_k f(z) + \frac{1}{2}\|z - z_k\|_2^2, \qquad \psi^n(z) := s_k h(z),$$

whose proximal operator is simply $\operatorname{prox}_{\tau \psi^n}(u) =
\operatorname{prox}_{(\tau s_k) h}(u)$. Whenever $\hat{m}_k \ge m$, $\psi^s$ is
$\frac{1}{2}$-strongly convex without requiring any prior bound on $m$.

### `PenalizedOracle` (Moreau Envelope Penalty & Augmented Lagrangian)

To enforce $\mathcal{A}(x) \in \mathcal{C}$ with penalty $c > 0$, multiplier
$p \in \mathcal{Y}$, and dampening parameter $\theta \in [0, 1)$, let
$\bar{p} = (1 - \theta) p$ and $w(x) = \mathcal{A}(x) + \bar{p}/c$. By the
**Moreau decomposition** of the indicator function $\delta_{\mathcal{C}}$, the
squared-distance penalty

$$\psi_c(x; \bar{p}) := \frac{c}{2}\big\|w(x) - \Pi_{\mathcal{C}}(w(x))\big\|_2^2 - \frac{1}{2c}\|\bar{p}\|_2^2$$

is **convex and continuously differentiable** with $c\|\mathcal{A}\|_{\text{op}}^2$-Lipschitz
gradient:

$$\nabla_x \psi_c(x; \bar{p}) = \mathcal{A}^*\!\big(\hat{p}(x)\big), \qquad \hat{p}(x) := c\big(w(x) - \Pi_{\mathcal{C}}(w(x))\big) \in N_{\mathcal{C}}\!\big(\Pi_{\mathcal{C}}(w(x))\big).$$

`PenalizedOracle` adds $\psi_c(x; \bar{p})$ to $f(x)$ while leaving $h(x)$ and
the weak-convexity constant $m$ of $f$ **completely unchanged** (because adding
a convex penalty cannot introduce negative curvature!). Setting $p = \text{None}$
recovers pure quadratic penalty methods (`QPAPD`, `QPAIPP`); maintaining $p_k$
with $\theta = 0$ gives classical proximal ALM (`IALM`, `IAIPAL`), and
$\theta \in (0, 1)$ gives dampened ALM (`AIDAL`).

---

## 4. Min-Max & Saddle-Point Oracles (`MinMaxOracle`)

`MinMaxOracle(coupling=..., prox_x=..., prox_y=...)` implements the composite
saddle-point model
$\mathcal{L}(x, y) = \Phi(x, y) + h_x(x) - h_y(y)$ for primal-dual pairs
$z = (x, y)$:

- **Coupling queries:** `phi_val(x, y)` returns $\Phi(x, y)$, `phi_val_grad(x, y)`
  returns $\big(\Phi(x, y), \nabla_x \Phi(x, y), \nabla_y \Phi(x, y)\big)$ via
  reverse-mode AD, and `grad_x(x, y)` / `grad_y(x, y)` expose partial gradients
  individually for alternating or Gauss-Seidel updates (`PDHG`).
- **Separable proximal steps:** `prox_x(u, tau)` and `prox_y(v, sigma)` evaluate
  $\operatorname{prox}_{\tau h_x}(u)$ and $\operatorname{prox}_{\sigma h_y}(v)$
  with optional `domain_project(x, y)` for initial iterates.
- **Full saddle-point objective:** `obj_val(x, y)` evaluates
  $\Phi(x, y) + h_x(x) - h_y(y)$ using the `nonsmooth_x` / `nonsmooth_y` value
  hooks when finite.

---

## 5. Spectral Oracles (`SpectralOracle` & `VectorSpectralSubOracle`)

`SpectralOracle(f1=..., f2_vec=..., prox_vec=..., domain_proj=...)` implements
the matrix spectral composite model
$\Phi(X) = f_1(X) + f_2^{\mathcal{V}}(\sigma(X)) + h^{\mathcal{V}}(\sigma(X))$
in two ways simultaneously:

1. **As a full matrix `CompositeOracleProtocol`:** evaluating thin SVDs so any
   general solver (`APD`, `AIPP`, `ECG`) can optimize a `SpectralProblem`.
2. **Via 1D singular-value reduction (`VectorSpectralSubOracle`):** at each
   outer iteration of `IAICG` or `DAICG`, given matrix center $Z_0$, step size
   $\lambda > 0$, and curvature shift $\xi \ge 0$, the solver computes a
   **single thin SVD**:

   $$Z_0 - \lambda \big(\nabla f_1(Z_0) - \xi Z_0\big) = P \operatorname{diag}(x_0) Q^\top$$

   and hands `VectorSpectralSubOracle` to the 1D accelerated composite gradient
   method (`ACG`) on $\mathbb{R}^r$:

   $$\Psi_\lambda(s) := \underbrace{\lambda \left(f_2^{\mathcal{V}}(s) + \frac{\xi}{2}\|s\|_2^2\right) + \frac{1}{4}\|s - x_0\|_2^2}_{\psi^s(s)} + \underbrace{\lambda h^{\mathcal{V}}(s) + \frac{1}{4}\|s - x_0\|_2^2}_{\psi^n(s)}.$$

   Every inner iteration operates purely on vectors of length $r = \min(p, q)$
   with **zero matrix SVDs**.

---

## 6. Exact Stationarity Refinement (`certificates.refine`)

How does Proxax compute an **exact** subgradient residual
$\hat{v} \in \nabla f(\hat{x}) + \partial h(\hat{x})$ when $h$ is nonsmooth and
may even be an indicator function with an unbounded normal cone?

Given any candidate iterate $y \in \operatorname{dom} h$,
`proxax.certificates.refine(oracle, y, lipschitz_init)` takes a single composite
gradient step with local smoothness constant $L > 0$ (found in closed form if
$M$ is known, or via backtracking so $f(\hat{x}) \le f(y) + \langle \nabla f(y),
\hat{x} - y \rangle + \frac{L}{2}\|\hat{x} - y\|_2^2$):

$$\hat{x} := \operatorname{prox}_{h / L}\!\left(y - \frac{1}{L}\nabla f(y)\right) = \operatorname*{argmin}_{u \in \mathcal{X}} \left\{ h(u) + \frac{L}{2}\left\|u - \left(y - \frac{1}{L}\nabla f(y)\right)\right\|_2^2 \right\}.$$

By the first-order optimality condition of this strongly convex minimization,
$0 \in \partial h(\hat{x}) + L\big(\hat{x} - y + \frac{1}{L}\nabla f(y)\big)$,
which rearranges to the **exact algebraic identity**:

$$L(y - \hat{x}) - \nabla f(y) \;\in\; \partial h(\hat{x}).$$

Adding $\nabla f(\hat{x})$ to both sides yields the certified stationarity
vector:

$$\hat{v} := \nabla f(\hat{x}) - \nabla f(y) + L(y - \hat{x}) \;\in\; \nabla f(\hat{x}) + \partial h(\hat{x}),$$

along with guaranteed composite descent $\phi(\hat{x}) \le \phi(y) - \frac{L}{2}\|\hat{x} - y\|_2^2$.

- **Constrained case:** Applying `refine` to `PenalizedOracle` yields
  $\hat{v} \in \nabla f(\hat{x}) + \mathcal{A}^*(\hat{p}(\hat{x})) + \partial h(\hat{x})$,
  where $\hat{p}(\hat{x}) = c\big(w(\hat{x}) - \Pi_{\mathcal{C}}(w(\hat{x}))\big)$
  automatically belongs to $N_{\mathcal{C}}(\Pi_{\mathcal{C}}(w(\hat{x})))$.
- **Min-max case (`refine_minmax`):** Takes a coupled primal-dual proximal
  step $(\hat{x}, \hat{y})$ with step sizes $(\tau, \sigma)$ from $(x, y)$ and
  recovers the exact saddle-point subgradient pair
  $\hat{v}_x \in \nabla_x \Phi(\hat{x}, \hat{y}) + \partial h_x(\hat{x})$ and
  $\hat{v}_y \in -\nabla_y \Phi(\hat{x}, \hat{y}) + \partial h_y(\hat{y})$.
- **Spectral case (`refine_icg`):** Performs the exact same refinement in 1D
  singular-value coordinates $\hat{z} \in \mathbb{R}^r$ and lifts the vector
  subgradient back to $\mathbb{R}^{p \times q}$ via $P \operatorname{diag}(\cdot) Q^\top$
  using a single matrix gradient evaluation $\nabla f_1(\hat{Z})$.
- **Zero-overhead certificates in `APD`:** Inner `PF.ACG` steps inside `APD`
  already have this exact composite-gradient form on the rescaled subproblem, so
  `APD` extracts $\hat{v}$ directly at every inner iteration with **zero extra
  oracle calls**.
