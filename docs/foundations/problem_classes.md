# Problem Classes & Optimality

Proxax solves five structured classes of finite-dimensional smooth and
nonsmooth optimization problems over **Euclidean PyTree spaces**. This page
defines the underlying geometry and curvature assumptions, and links to each
problem class formulation and stationarity / KKT certificate.

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

$$-\frac{m}{2}\|y - x\|_2^2 \le f(y) - f(x) - \langle \nabla f(x), y - x \rangle \le \frac{M}{2}\|y - x\|_2^2.$$

- When $m = 0$, $f$ is **convex**.
- When $m > 0$, $f$ is **$m$-weakly convex** (so $x \mapsto f(x) + \frac{m}{2}\|x\|_2^2$ is convex), while $M \ge m$ may be orders of magnitude larger.
- Most first-order methods in the literature assume a single symmetric bound
  $L = \max(m, M)$ and scale step sizes by $1/L$. Proxax's adaptive proximal
  solvers (`APD`, `AIPP`, `QPAPD`, `QPAIPP`, `IAIPAL`, `SProxALM`) explicitly
  exploit $m \ll M$ to take larger proximal steps $\lambda \approx 1/(2m)$
  while running accelerated convex solvers inside each subproblem.

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

## Problem Classes

### 1. Unconstrained Composite (`CompositeProblem`)

$$\min_{x \in \mathcal{X}} \; \phi(x) := f(x) + h(x)$$

Here $f$ is differentiable and $(m, M)$-smooth/weakly convex on
$\operatorname{dom} h$, and $h$ admits a tractable proximal operator
$\operatorname{prox}_{\tau h}(u) := \operatorname*{argmin}_z \{ h(z) + \frac{1}{2\tau}\|z - u\|_2^2 \}$.
Simple direct sets $x \in K$ (boxes, balls, simplices, cones) are handled
directly by setting $h = \delta_K$.

Given `opt_tol` $\rho > 0$, unconstrained solvers return a certified
$\rho$-stationary pair $(\hat{x}, \hat{v})$ (`sol.x`, `sol.subgrad`):

$$\hat{v} \in \nabla f(\hat{x}) + \partial h(\hat{x}), \qquad \|\hat{v}\| \le \rho.$$

### 2. Set- and Cone-Constrained (`CompositeProblem`)

$$\min_{x \in \mathcal{X}} \; f(x) + h(x) \quad \text{s.t.} \quad \mathcal{A}(x) \in \mathcal{C}$$

where $\mathcal{A} : \mathcal{X} \to \mathcal{Y}$ is a linear map with adjoint
$\mathcal{A}^*$ and $\mathcal{C} \subseteq \mathcal{Y}$ is a closed convex set
equipped with Euclidean projector $\Pi_{\mathcal{C}}$:

| Constraint Type | Set $\mathcal{C}$ | Projector $\Pi_{\mathcal{C}}(w)$ | Normal Cone $N_{\mathcal{C}}(z)$ |
| :--- | :--- | :--- | :--- |
| Equality $\mathcal{A}(x) = b$ | $\{b\}$ (`Singleton`) | $b$ | $\mathcal{Y}$ |
| Upper $\mathcal{A}(x) \le u$ | $(-\infty, u]$ (`UpperBound`) | $\min(w, u)$ | $\{p \ge 0 : p_i (z_i - u_i) = 0\}$ |
| Lower $\mathcal{A}(x) \ge l$ | $[l, +\infty)$ (`LowerBound`) | $\max(w, l)$ | $\{p \le 0 : p_i (z_i - l_i) = 0\}$ |
| Box $l \le \mathcal{A}(x) \le u$ | $[l, u]$ (`BoxSet`) | $\operatorname{clip}(w, l, u)$ | Sign-complementary multipliers |
| Cones ($\text{SOC}$, $\text{Exp}$, $\text{PSD}$) | $\mathcal{K}$ | Exact cone projection | $-\mathcal{K}^*$ with complementarity |

Given $(\rho, \eta)$ (`opt_tol`, `feas_tol`), constrained solvers return
$(\hat{x}, \hat{p}, \hat{v})$ (`sol.x`, `sol.dual`, `sol.subgrad`) with
$\hat{p} \in N_{\mathcal{C}}(\hat{z})$ exact for some $\hat{z} \in \mathcal{C}$:

$$\hat{v} \in \nabla f(\hat{x}) + \partial h(\hat{x}) + \mathcal{A}^*(\hat{p}), \qquad \|\hat{v}\| \le \rho, \qquad \|\mathcal{A}(\hat{x}) - \Pi_{\mathcal{C}}(\mathcal{A}(\hat{x}))\| \le \eta.$$

### 3. Block-Constrained Splitting (`BlockProblem`)

$$\min_{x_1, \dots, x_K} \; \sum_{i=1}^K \big(f_i(x_i) + h_i(x_i)\big) \qquad \text{s.t.} \qquad \sum_{i=1}^K \mathcal{A}_i(x_i) = b.$$

Block ADMM solvers (`ADMM`, `DPADMM`, `PAdaptADMM`) return blocks
$\hat{x} = (\hat{x}_1, \dots, \hat{x}_K)$, dual $\hat{y}$, and subgradients
$\hat{v} = (\hat{v}_1, \dots, \hat{v}_K)$ satisfying:

$$\hat{v}_i \in \nabla f_i(\hat{x}_i) + \mathcal{A}_i^*(\hat{y}) + \partial h_i(\hat{x}_i), \qquad \|\hat{v}\|_2 \le \rho, \qquad \left\|\sum_{i=1}^K \mathcal{A}_i(\hat{x}_i) - b\right\|_2 \le \eta.$$

### 4. Min-Max & Saddle-Point (`MinMaxProblem`)

$$\min_{x \in \mathcal{X}} \max_{y \in \mathcal{Y}} \; \Phi(x, y) + h(x) - g(y)$$

Min-max solvers (`ExtraGradient`, `PDHG`, `ProxPointMinMax`) return $(\hat{x}, \hat{y})$
and $(\hat{v}_x, \hat{v}_y)$ satisfying:

$$\hat{v}_x \in \nabla_x \Phi(\hat{x}, \hat{y}) + \partial h(\hat{x}), \qquad \hat{v}_y \in -\nabla_y \Phi(\hat{x}, \hat{y}) + \partial g(\hat{y}), \qquad \sqrt{\|\hat{v}_x\|_2^2 + \|\hat{v}_y\|_2^2} \le \rho.$$

### 5. Matrix Spectral Composite (`SpectralProblem`)

$$\min_{X \in \mathbb{R}^{p \times q}} \; \Phi(X) := f_1(X) + (f_2^{\mathcal{V}} \circ \sigma)(X) + (h^{\mathcal{V}} \circ \sigma)(X)$$

where $\sigma(X) \in \mathbb{R}_+^r$ ($r = \min(p, q)$) are the singular values
of $X$, and $f_2^{\mathcal{V}}, h^{\mathcal{V}}$ are **absolutely symmetric**.
By **Lewis's Theorem**, gradients and proximal steps of orthogonally invariant
terms reduce to $r$-dimensional vector operations on $\sigma(X)$ using a single
thin SVD $X = P \operatorname{diag}(s) Q^\top$. Spectral solvers (`DAICG`,
`IAICG`) also shift quadratic curvature $\xi \ge 0$ between $f_1$ and
$f_2^{\mathcal{V}}$ to convexify $f_2^{\mathcal{V}}$ and certify:

$$\hat{V} \in \nabla f_1(\hat{Z}) + \nabla (f_2^{\mathcal{V}} \circ \sigma)(\hat{Z}) + \partial (h^{\mathcal{V}} \circ \sigma)(\hat{Z}), \qquad \|\hat{V}\|_F \le \rho.$$
