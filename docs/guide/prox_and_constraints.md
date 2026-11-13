# Proximal Operators & Constraints

`proxax` gives you two ways to add nonsmooth structure or constraints to a
smooth objective $f(x)$:

1. **`proximal=...` (`proxax.prox`):** For convex regularizers (like $\ell_1$,
   elastic net, or nuclear norm) and **simple direct sets** on $x$ (like
   `NonNegative`, `Box`, `Simplex`, `L2Ball`, or `L1Ball`) that have a fast
   closed-form proximal operator or projection.
2. **`constraints=...` (`proxax.constraints`):** For **coupled constraints** of
   the form $\mathcal{A}(x) \in \mathcal{C}$ (like $A x = b$, $A x \le u$,
   consensus $X - Y = 0$, or second-order cones), handled via penalty or
   augmented Lagrangian solvers.

:::{tip}
**Rule of thumb:** Put direct variable bounds (such as $0 \le x \le 1$ or
$x \in \Delta^n$) in `proximal=...` whenever you can, and reserve
`constraints=...` for linear/cone relationships that couple variables together!
For example, $\{x \in [0, 1]^n : A x = b\}$ runs much faster with
`proximal=prox.Box(lower=0.0, upper=1.0)` + `constraints=constraints.equality(A, b)`
than putting both in `constraints`.
:::

---

## 1. Built-In Proximal Operators (`proxax.prox`)

### Norm Regularizers

| Operator | Mathematical Formula $h(x)$ | Basic Usage |
| :--- | :--- | :--- |
| `prox.Zero()` | $0$ | `prox.Zero()` (default when `proximal` is omitted) |
| `prox.L1(weight=1.0)` | $w \|x\|_1$ | `prox.L1(weight=0.1)` (elementwise soft-thresholding) |
| `prox.L2(weight=1.0)` | $w \|x\|_2$ | `prox.L2(weight=0.5)` (group / block soft-thresholding) |
| `prox.L2Squared(weight=1.0)` | $\frac{w}{2}\|x\|_2^2$ | `prox.L2Squared(weight=1.0)` ($w$-strongly convex shrinkage) |
| `prox.ElasticNet(l1_weight=1.0, l2_weight=1.0)` | $w_1 \|x\|_1 + \frac{w_2}{2}\|x\|_2^2$ | `prox.ElasticNet(l1_weight=0.1, l2_weight=0.5)` |
| `prox.NuclearNorm(weight=1.0)` | $w \|\sigma(X)\|_1$ | `prox.NuclearNorm(weight=0.2)` (matrix singular-value thresholding) |

```python
import jax.numpy as jnp
from proxax import prox

op = prox.ElasticNet(l1_weight=0.2, l2_weight=1.0)
x_shrunk = op.prox(jnp.array([1.5, -0.1, 0.05]), step_size=0.5)
print(x_shrunk)  # [0.93333334, 0.0, 0.0]
```

### Set Indicators & Box-Restricted Norms

Indicator operators enforce $x \in K$ strictly at every step:

| Operator | Feasible Set $K$ | Basic Usage |
| :--- | :--- | :--- |
| `prox.NonNegative()` | $\{x : x \ge 0\}$ | `prox.NonNegative()` |
| `prox.Box(lower=..., upper=...)` | $\{x : l \le x \le u\}$ | `prox.Box(lower=0.0, upper=1.0)` (accepts scalars or matching PyTrees) |
| `prox.BoxL1(l1_weight=..., lower=..., upper=...)` | $w \|x\|_1$ on $\{x : l \le x \le u\}$ | `prox.BoxL1(l1_weight=0.1, lower=-1.0, upper=1.0)` |
| `prox.Simplex(scale=1.0)` | $\{x \ge 0 : \sum_i x_i = s\}$ | `prox.Simplex(scale=1.0)` (probability simplex) |
| `prox.L2Ball(radius=1.0)` | $\{x : \|x\|_2 \le r\}$ | `prox.L2Ball(radius=2.5)` |
| `prox.L1Ball(radius=1.0)` | $\{x : \|x\|_1 \le r\}$ | `prox.L1Ball(radius=1.0)` |

### PyTree & Shifted Combinators

- **`prox.MultiProx(operators)`:** Applies a different `ProxOperator` to each
  key or branch of a PyTree variable $x = (x_1, \dots, x_k)$, representing the
  separable sum $h(x) = \sum_{i=1}^k h_i(x_i)$:

  ```python
  import jax.numpy as jnp
  from proxax import prox

  multi = prox.MultiProx({
      "weights": prox.L1(weight=0.05),
      "probs": prox.Simplex(scale=1.0),
      "bias": prox.Zero(),
  })
  x = {
      "weights": jnp.array([0.4, -0.01]),
      "probs": jnp.array([0.2, 0.9]),
      "bias": jnp.asarray(1.2),
  }
  print(multi.prox(x, step_size=1.0))
  ```

- **`prox.Shifted(base=..., center=..., reg=...)`:** Adds a quadratic anchor
  $\frac{\text{reg}}{2}\|x - \text{center}\|_2^2$ to any `base` `ProxOperator`
  with an exact closed-form proximal step.

---

## 2. Linear & Cone Constraints (`proxax.constraints`)

### Convenience Constraint Builders

Every builder in `proxax.constraints` takes a **linear operator** (either a 2D
`jax.Array` matrix `A` or any linear Python function `lambda x: ...`) and target
bounds:

```python
import jax.numpy as jnp
from proxax import constraints

a_mat = jnp.array([[1.0, 2.0, -1.0], [0.5, -0.5, 1.0]])
b_vec = jnp.array([1.0, 0.0])

# 1. Equality: A @ x == b
c_eq = constraints.equality(a_mat, rhs=b_vec)

# 2. Upper inequality: A @ x <= b
c_le = constraints.less_equal(a_mat, rhs=b_vec)

# 3. Lower inequality: A @ x >= 0
c_ge = constraints.greater_equal(a_mat, rhs=0.0)

# 4. Two-sided linear box: -1 <= A @ x <= 1
c_box = constraints.box(a_mat, lower=-1.0, upper=1.0)

# 5. Second-order (Lorentz) cone: ||[x1, x2]||_2 <= x0
c_soc = constraints.in_set(lambda x: x, constraints.SecondOrderCone())
```

### Matrix-Free PyTree Linear Operators

Because `constraints` accepts arbitrary linear Python functions, you never have
to materialize huge Kronecker or block matrices for matrix variables or nested
dictionaries. `proxax` computes the adjoint $\mathcal{A}^*$ automatically via
`jax.linear_transpose`:

```python
import jax.numpy as jnp
import proxax
from proxax import constraints
from proxax import prox

# Consensus constraint X - Y == 0 on a dict of two 4x4 matrices:
problem = proxax.CompositeProblem(
    smooth=lambda z: 0.5 * jnp.sum((z["x"] - 1.0) ** 2),
    proximal=prox.MultiProx({
        "x": prox.NuclearNorm(weight=0.5),
        "y": prox.Box(lower=0.0, upper=0.8),
    }),
    constraints=constraints.equality(lambda z: z["x"] - z["y"], rhs=0.0),
)

z0 = {"x": jnp.zeros((4, 4)), "y": jnp.zeros((4, 4))}
sol = proxax.solve(problem, z0, method="ialm", opt_tol=1e-5, feas_tol=1e-5)
```

If you already have a fast custom transpose or a known bound on the squared
operator norm $\|\mathcal{A}\|_{\text{op}}^2$, pass `rmatvec=...` or
`operator_norm_sq=...` to any builder (e.g.,
`constraints.equality(fn, rhs=0.0, operator_norm_sq=2.0)`).

### Combining Multiple Constraints in a `dict` or `tuple`

Need equality, inequality, and cone constraints in the same problem? Pass a
Python `dict`, `tuple`, or `list` of `Constraint` objects directly to
`CompositeProblem(constraints=...)`. `proxax` automatically combines them into a
`ProductConstraint` and returns `sol.dual` with the **exact same keys/structure**:

```python
import jax.numpy as jnp
import proxax
from proxax import constraints
from proxax import prox

problem = proxax.CompositeProblem(
    smooth=lambda x: 0.5 * jnp.sum((x - jnp.array([2.0, 0.5, -0.5])) ** 2),
    proximal=prox.NonNegative(),
    constraints={
        "budget": constraints.equality(lambda x: jnp.sum(x), rhs=1.0),
        "cap": constraints.less_equal(lambda x: x[0] + x[1], rhs=0.8),
    },
)

sol = proxax.solve(problem, jnp.ones(3) / 3.0, method="ialm", opt_tol=1e-5, feas_tol=1e-5)
print("Primal x:       ", sol.x)
print("Budget dual:    ", sol.dual["budget"])
print("Cap dual (>=0): ", sol.dual["cap"])
```
