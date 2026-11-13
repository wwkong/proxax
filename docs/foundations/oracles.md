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

## 3. Internal Solver Oracles & Exact Certificates

For the algebraic subproblem wrappers (`ProxRegularizedOracle`,
`RescaledProxOracle`, `PenalizedOracle`, `MinMaxOracle`, `SpectralOracle`) and
exact subdifferential refinement (`certificates.refine`), see
{doc}`../appendices/oracles_and_certificates`.

