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

```{toctree}
:maxdepth: 1

problem_classes/unconstrained
problem_classes/constrained
problem_classes/block
problem_classes/minmax
problem_classes/spectral
```
