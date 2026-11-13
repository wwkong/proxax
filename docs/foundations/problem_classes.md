# Problem Classes & Optimality

Proxax solves four structured classes of finite-dimensional smooth and
nonsmooth optimization problems over **Euclidean PyTree spaces**. This page
defines the underlying geometry, curvature assumptions, problem formulations,
and exact stationarity / KKT certificates returned by every solver.

---

## Euclidean Geometry on JAX PyTrees

Let $\mathcal{X}$ be a finite-dimensional real vector space represented in JAX
as a **PyTree** of floating-point arrays $x = (x^{(1)}, \dots, x^{(L)})$.
Proxax equips $\mathcal{X}$ with the standard leaf-wise inner product and
Euclidean ($\ell_2$) norm:

$$\langle x, y \rangle := \sum_{\ell=1}^{L} \langle x^{(\ell)}, y^{(\ell)} \rangle_F, \qquad \|x\|_2 := \sqrt{\langle x, x \rangle}, \qquad \|x\|_\infty := \max_{\ell=1,\dots,L} \|x^{(\ell)}\|_\infty.$$

When `norm_kind="l2"` (the default), all stationarity and feasibility tolerances
are measured in $\|\cdot\|_2$; setting `norm_kind="linf"` switches stopping
tests to $\|\cdot\|_\infty$.

---

## Smoothness, Weak Convexity & Convex Subdifferentials

Throughout Proxax, composite objectives split into a differentiable
(possibly nonconvex) function $f$ and a proper, closed, convex function $h$:

### Asymmetric Curvature Bounds $(m, M)$ on $f$

Let $f : \mathcal{X} \to \mathbb{R}$ be continuously differentiable on an open
neighborhood of $\operatorname{cl}(\operatorname{dom} h)$. We say that $f$ has
**lower curvature (weak convexity) $m \ge 0$** and **upper curvature
(smoothness) $M \ge 0$** on $\operatorname{dom} h$ if, for all
$x, y \in \operatorname{dom} h$:

$$-\frac{m}{2}\|y - x\|_2^2 \;\le\; f(y) - f(x) - \langle \nabla f(x), y - x \rangle \;\le\; \frac{M}{2}\|y - x\|_2^2.$$

- **Convex special case ($m = 0$):** When $m = 0$, the left-hand inequality
  reduces to convexity $f(y) \ge f(x) + \langle \nabla f(x), y - x \rangle$.
- **Weakly convex regime ($m > 0$):** Adding $\frac{m}{2}\|x\|_2^2$ cancels the
  negative curvature of $f$, making $x \mapsto f(x) + \frac{m}{2}\|x - z\|_2^2$
  convex for every center $z \in \mathcal{X}$ (and strongly convex whenever the
  regularization weight exceeds $m$).
- **Why track $(m, M)$ separately?** For many nonconvex objectives (e.g.,
  indefinite QPs, $\tanh$ margin losses, or concave sparsity penalties), the
  negative curvature $m$ is much smaller than the positive smoothness $M$
  ($m \ll M$). Proximal-point methods need only add $\approx m$ of quadratic
  regularization rather than $M$, taking much larger outer steps.
- **Parameter-free operation:** Supplying $(m, M)$ is always **optional** in
  Proxax; adaptive solvers (`APD`, `AdapFISTA`, `AdapAPG`, `UPFAG`, `ECG`,
  `QPAPD`, `IALM`) estimate curvatures dynamically from function and gradient
  evaluations.

### Convex Regularizer $h$ and Its Subdifferential

A function $h : \mathcal{X} \to (-\infty, +\infty]$ is **proper, closed, and
$\mu$-strongly convex** ($\mu \ge 0$) with nonempty effective domain
$\operatorname{dom} h := \{x \in \mathcal{X} : h(x) < +\infty\}$ if
$x \mapsto h(x) - \frac{\mu}{2}\|x\|_2^2$ is proper, lower semicontinuous, and
convex. Its **convex subdifferential** at $x \in \operatorname{dom} h$ is:

$$\partial h(x) := \left\{ s \in \mathcal{X} \;:\; h(y) \ge h(x) + \langle s, y - x \rangle + \frac{\mu}{2}\|y - x\|_2^2 \quad \forall y \in \mathcal{X} \right\}.$$

When $h = \delta_K$ is the $\{0, +\infty\}$ **indicator function** of a nonempty
closed convex set $K \subseteq \mathcal{X}$, $\operatorname{dom} h = K$ and
$\partial \delta_K(x)$ coincides with the **normal cone** of $K$ at $x$:

$$N_K(x) := \big\{ s \in \mathcal{X} \;:\; \langle s, y - x \rangle \le 0 \quad \forall y \in K \big\}.$$

---

## Class 1: Unconstrained Composite Optimization (`CompositeProblem`)

### Formulation

$$\min_{x \in \mathcal{X}} \; \phi(x) := f(x) + h(x)$$

where $f$ is differentiable and $(m, M)$-smooth/weakly convex on
$\operatorname{dom} h$, and $h$ is proper, closed, $\mu$-strongly convex, and
admits a tractable proximal operator:

$$\operatorname{prox}_{\tau h}(u) := \operatorname*{argmin}_{z \in \mathcal{X}} \left\{ h(z) + \frac{1}{2\tau}\|z - u\|_2^2 \right\}, \qquad \tau > 0.$$

Note that "unconstrained" here means **no coupling linear operator
$\mathcal{A}(x) \in \mathcal{C}$ is present**: simple geometric constraints
$x \in K$ (boxes, balls, non-negative orthants, or simplices) are handled
directly and exactly by setting $h = \delta_K$.

### Certified $\rho$-Stationarity

A point $x^\star \in \operatorname{dom} h$ is **first-order stationary** if
$0 \in \nabla f(x^\star) + \partial h(x^\star)$. Given tolerance $\rho > 0$, an
unconstrained solver in Proxax returns a **certified $\rho$-stationary pair**
$(\hat{x}, \hat{v}) \in \operatorname{dom} h \times \mathcal{X}$ satisfying:

$$\hat{v} \in \nabla f(\hat{x}) + \partial h(\hat{x}), \qquad \|\hat{v}\| \le \rho.$$

In `Solution`, `sol.x` holds $\hat{x}$, `sol.subgrad` holds $\hat{v}$, and
`sol.stats.opt_residual` equals $\|\hat{v}\|$.

---

## Class 2: Set- and Cone-Constrained Optimization (`CompositeProblem`)

### Formulation

$$\min_{x \in \mathcal{X}} \; f(x) + h(x) \quad \text{s.t.} \quad \mathcal{A}(x) \in \mathcal{C}$$

where:

- $\mathcal{A} : \mathcal{X} \to \mathcal{Y}$ is a linear map between Euclidean
  PyTree spaces $(\mathcal{X}, \langle \cdot, \cdot \rangle)$ and
  $(\mathcal{Y}, \langle \cdot, \cdot \rangle)$, with **adjoint**
  $\mathcal{A}^* : \mathcal{Y} \to \mathcal{X}$ uniquely defined by:

  $$\langle \mathcal{A}(x), p \rangle = \langle x, \mathcal{A}^*(p) \rangle \qquad \forall x \in \mathcal{X},\; p \in \mathcal{Y}.$$

- $\mathcal{C} \subseteq \mathcal{Y}$ is a nonempty closed convex set (or a
  Cartesian product $\mathcal{C}_1 \times \dots \times \mathcal{C}_k$ of convex
  sets) equipped with its exact **Euclidean projector**:

  $$\Pi_{\mathcal{C}}(w) := \operatorname*{argmin}_{z \in \mathcal{C}} \; \frac{1}{2}\|z - w\|_2^2.$$

This single set-inclusion abstraction $\mathcal{A}(x) \in \mathcal{C}$ unifies:

| Constraint Type | Set $\mathcal{C}$ | Projector $\Pi_{\mathcal{C}}(w)$ | Normal Cone $N_{\mathcal{C}}(z)$ |
| :--- | :--- | :--- | :--- |
| Linear equality $\mathcal{A}(x) = b$ | $\{b\}$ (`Singleton(b)`) | $b$ | $\mathcal{Y}$ (free multipliers) |
| Inequality $\mathcal{A}(x) \le u$ | $(-\infty, u]$ (`UpperBound(u)`) | $\min(w, u)$ | $\{p \ge 0 : p_i (z_i - u_i) = 0\}$ |
| Inequality $\mathcal{A}(x) \ge l$ | $[l, +\infty)$ (`LowerBound(l)`) | $\max(w, l)$ | $\{p \le 0 : p_i (z_i - l_i) = 0\}$ |
| Two-sided bounds $l \le \mathcal{A}(x) \le u$ | $[l, u]$ (`BoxSet`) | $\operatorname{clip}(w, l, u)$ | Sign-complementary multipliers |
| Second-order cone $\|(u_1, \dots, u_d)\|_2 \le t$ | $\mathcal{K}_{\text{SOC}}$ (`SecondOrderCone`) | Closed-form Lorentz cone projection | $-\mathcal{K}_{\text{SOC}}$ with complementarity |

### Certified $(\rho, \eta)$-KKT Conditions

A primal-dual pair $(x^\star, p^\star) \in \operatorname{dom} h \times \mathcal{Y}$
is a **KKT point** if:

$$0 \in \nabla f(x^\star) + \partial h(x^\star) + \mathcal{A}^*(p^\star), \qquad \mathcal{A}(x^\star) \in \mathcal{C}, \qquad p^\star \in N_{\mathcal{C}}\!\big(\mathcal{A}(x^\star)\big).$$

Given optimality tolerance $\rho > 0$ (`opt_tol`) and feasibility tolerance
$\eta > 0$ (`feas_tol`), constrained solvers in Proxax return a **certified
$(\rho, \eta)$-KKT triplet** $(\hat{x}, \hat{p}, \hat{v})$ satisfying:

1. **Domain membership:** $\hat{x} \in \operatorname{dom} h$.
2. **Approximate primal feasibility:**

   $$\big\|\mathcal{A}(\hat{x}) - \Pi_{\mathcal{C}}\!\big(\mathcal{A}(\hat{x})\big)\big\| \le \eta.$$

3. **Normal-cone dual inclusion & Lagrangian stationarity:** there exists a
   projected point $\hat{z} = \Pi_{\mathcal{C}}(w) \in \mathcal{C}$ such that
   $\hat{p} \in N_{\mathcal{C}}(\hat{z})$ holds **exactly** and:

   $$\hat{v} \in \nabla f(\hat{x}) + \partial h(\hat{x}) + \mathcal{A}^*(\hat{p}), \qquad \|\hat{v}\| \le \rho.$$

In `Solution`, `sol.x` holds $\hat{x}$, `sol.dual` holds $\hat{p}$,
`sol.subgrad` holds $\hat{v}$, `sol.stats.opt_residual` equals $\|\hat{v}\|$,
and `sol.stats.feas_residual` equals
$\|\mathcal{A}(\hat{x}) - \Pi_{\mathcal{C}}(\mathcal{A}(\hat{x}))\|$.

---

## Class 3: Block-Constrained Composite Optimization (`BlockProblem`)

### Formulation

Given $K$ separable `BlockTerm` blocks $\{(f_i, h_i, \mathcal{A}_i)\}_{i=1}^K$
and right-hand side $b$, a `BlockProblem` solves:

$$\min_{x_1, \dots, x_K} \; \sum_{i=1}^K \big(f_i(x_i) + h_i(x_i)\big) \qquad \text{s.t.} \qquad \sum_{i=1}^K \mathcal{A}_i(x_i) = b.$$

### Certified Block KKT $(\rho, \eta)$-Stationarity

Given tolerances $\rho, \eta > 0$ (`opt_tol`, `feas_tol`), `ADMM` returns
blocks $\hat{x} = (\hat{x}_1, \dots, \hat{x}_K)$, dual multiplier $\hat{y}$,
and subgradients $\hat{v} = (\hat{v}_1, \dots, \hat{v}_K)$ satisfying:

$$\hat{v}_i \in \nabla f_i(\hat{x}_i) + \mathcal{A}_i^*(\hat{y}) + \partial h_i(\hat{x}_i), \qquad \|\hat{v}\|_2 \le \rho, \qquad \left\|\sum_{i=1}^K \mathcal{A}_i(\hat{x}_i) - b\right\|_2 \le \eta.$$

---

## Class 4: Min-Max & Saddle-Point Optimization (`MinMaxProblem`)

### Formulation

Let $\mathcal{X}$ and $\mathcal{Y}$ be Euclidean PyTree spaces. A
`MinMaxProblem` solves:

$$\min_{x \in \mathcal{X}} \max_{y \in \mathcal{Y}} \; \Phi(x, y) + h(x) - g(y)$$

where $\Phi : \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$ is differentiable
(possibly nonconvex-concave or weakly-convex–weakly-concave) and $h$, $g$ are
proper, closed, convex, and proximable regularizers or set indicators.

### Certified Primal-Dual $\rho$-Stationarity

Given tolerance $\rho > 0$ (`opt_tol`), min-max solvers (`ExtraGradient`,
`PDHG`, `ProxPointMinMax`) return a certified primal-dual pair
$(\hat{x}, \hat{y})$ and subgradient pair $(\hat{v}_x, \hat{v}_y)$ satisfying:

$$\hat{v}_x \in \nabla_x \Phi(\hat{x}, \hat{y}) + \partial h(\hat{x}), \qquad \hat{v}_y \in -\nabla_y \Phi(\hat{x}, \hat{y}) + \partial g(\hat{y}), \qquad \sqrt{\|\hat{v}_x\|_2^2 + \|\hat{v}_y\|_2^2} \le \rho.$$

---

## Class 5: Matrix Spectral Composite Optimization (`SpectralProblem`)

### Formulation

Let $\mathcal{X} = \mathbb{R}^{p \times q}$ with $r := \min(p, q)$, and let
$\sigma : \mathbb{R}^{p \times q} \to \mathbb{R}_+^r$ map a matrix $X$ to its
singular values $\sigma_1(X) \ge \dots \ge \sigma_r(X) \ge 0$. A
`SpectralProblem` solves:

$$\min_{X \in \mathbb{R}^{p \times q}} \; \Phi(X) := f_1(X) + (f_2^{\mathcal{V}} \circ \sigma)(X) + (h^{\mathcal{V}} \circ \sigma)(X)$$

where:

1. $f_1 : \mathbb{R}^{p \times q} \to \mathbb{R}$ is a general differentiable
   matrix objective with curvature constants $(m_1, M_1)$ (for example,
   $\frac{1}{2}\|P_\Omega(X - M_{\text{obs}})\|_F^2$).
2. $f_2^{\mathcal{V}} : \mathbb{R}^r \to \mathbb{R}$ is a differentiable
   **absolutely symmetric** vector function with curvature constants
   $(m_2, M_2)$, meaning $f_2^{\mathcal{V}}(\epsilon \odot \Pi s) =
   f_2^{\mathcal{V}}(s)$ for every sign vector $\epsilon \in \{\pm 1\}^r$ and
   permutation matrix $\Pi$ (for example, smooth concave log-sum, SCAD, or MCP
   singular-value components).
3. $h^{\mathcal{V}} : \mathbb{R}^r \to (-\infty, +\infty]$ is a proper, closed,
   convex, **absolutely symmetric** proximable vector function (for example,
   $\mu \|s\|_1$ for the nuclear norm $\mu \|X\|_*$, optionally restricted to a
   Frobenius ball $\|s\|_2 \le R$).

### Spectral Calculus & Curvature Redistribution

By **Lewis's Theorem** for orthogonally invariant functions, if
$X = P \operatorname{diag}(s) Q^\top$ is a thin SVD of $X$
($P \in \mathbb{R}^{p \times r}$, $Q \in \mathbb{R}^{q \times r}$ with
orthonormal columns and $s = \sigma(X)$), then:

$$\nabla (f_2^{\mathcal{V}} \circ \sigma)(X) = P \operatorname{diag}\!\big(\nabla f_2^{\mathcal{V}}(s)\big) Q^\top, \qquad P \operatorname{diag}\!\big(\partial h^{\mathcal{V}}(s)\big) Q^\top \subseteq \partial (h^{\mathcal{V}} \circ \sigma)(X),$$

and the matrix proximal operator reduces to a 1D vector proximal step on $s$:

$$\operatorname{prox}_{\tau (h^{\mathcal{V}} \circ \sigma)}(X) = P \operatorname{diag}\!\big(\operatorname{prox}_{\tau h^{\mathcal{V}}}(s)\big) Q^\top.$$

Moreover, because $\|X\|_F^2 = \|\sigma(X)\|_2^2$, `IAICG` and `DAICG` can shift
any scalar $\xi \ge 0$ of quadratic curvature between the matrix term and the
singular-value term without changing the total objective $\Phi(X)$:

$$f_1^\xi(X) := f_1(X) - \frac{\xi}{2}\|X\|_F^2, \qquad f_2^{\mathcal{V},\xi}(s) := f_2^{\mathcal{V}}(s) + \frac{\xi}{2}\|s\|_2^2.$$

Choosing $\xi = m_2$ convexifies $f_2^{\mathcal{V},\xi}$ completely! Each outer
iteration linearizes $f_1^\xi$ at $Z_0$, computes **a single thin SVD** of the
matrix gradient step, solves a cheap $r$-dimensional vector subproblem for the
new singular values, and constructs an exact matrix certificate:

$$\hat{V} \in \nabla f_1(\hat{Z}) + \nabla (f_2^{\mathcal{V}} \circ \sigma)(\hat{Z}) + \partial (h^{\mathcal{V}} \circ \sigma)(\hat{Z}), \qquad \|\hat{V}\|_F \le \rho.$$
