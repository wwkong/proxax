# Appendix C: Internal Oracles & Exact Certificates

This appendix documents the algebraic oracle wrappers in `proxax.oracle` and
the exact subdifferential refinement identities in `proxax.certificates`.

---

## 1. Composite Oracles & Subproblem Transformations

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

## 2. Min-Max & Spectral Oracles (`MinMaxOracle`, `SpectralOracle`)

- **`MinMaxOracle(coupling=..., prox_x=..., prox_y=...)`** implements
  $\mathcal{L}(x, y) = \Phi(x, y) + h_x(x) - h_y(y)$, exposing `phi_val`,
  `phi_val_grad`, partial gradients `grad_x` / `grad_y` (`PDHG`), and separable
  proximal steps `prox_x` / `prox_y`.
- **`SpectralOracle(f1=..., f2_vec=..., prox_vec=..., domain_proj=...)`**
  implements $\Phi(X) = f_1(X) + f_2^{\mathcal{V}}(\sigma(X)) +
  h^{\mathcal{V}}(\sigma(X))$. At each outer iteration of `IAICG` or `DAICG`,
  the solver computes a single thin SVD

  $$Z_0 - \lambda \big(\nabla f_1(Z_0) - \xi Z_0\big) = P \operatorname{diag}(x_0) Q^\top$$

  and hands `VectorSpectralSubOracle` to the 1D `ACG` inner solver on
  $\mathbb{R}^r$ ($r = \min(p, q)$) with zero inner matrix SVDs:

  $$\Psi_\lambda(s) := \underbrace{\lambda \left(f_2^{\mathcal{V}}(s) + \frac{\xi}{2}\|s\|_2^2\right) + \frac{1}{4}\|s - x_0\|_2^2}_{\psi^s(s)} + \underbrace{\lambda h^{\mathcal{V}}(s) + \frac{1}{4}\|s - x_0\|_2^2}_{\psi^n(s)}.$$

---

## 3. Exact Stationarity Refinement (`certificates.refine`)

Given any candidate iterate $y \in \operatorname{dom} h$,
`proxax.certificates.refine(oracle, y, lipschitz_init)` takes a single composite
gradient step with local smoothness constant $L > 0$:

$$\hat{x} := \operatorname{prox}_{h / L}\!\left(y - \frac{1}{L}\nabla f(y)\right) = \operatorname*{argmin}_{u \in \mathcal{X}} \left\{ h(u) + \frac{L}{2}\left\|u - \left(y - \frac{1}{L}\nabla f(y)\right)\right\|_2^2 \right\}.$$

By first-order optimality, $L(y - \hat{x}) - \nabla f(y) \in \partial h(\hat{x})$.
Adding $\nabla f(\hat{x})$ yields the certified stationarity vector:

$$\hat{v} := \nabla f(\hat{x}) - \nabla f(y) + L(y - \hat{x}) \;\in\; \nabla f(\hat{x}) + \partial h(\hat{x}),$$

along with guaranteed composite descent $\phi(\hat{x}) \le \phi(y) - \frac{L}{2}\|\hat{x} - y\|_2^2$.

- **Constrained (`PenalizedOracle`):** `refine` yields $\hat{v} \in \nabla f(\hat{x}) + \mathcal{A}^*(\hat{p}(\hat{x})) + \partial h(\hat{x})$ with $\hat{p}(\hat{x}) \in N_{\mathcal{C}}(\Pi_{\mathcal{C}}(w(\hat{x})))$.
- **Min-max (`refine_minmax`):** Recovers $(\hat{v}_x, \hat{v}_y)$ from a coupled primal-dual proximal step.
- **Block (`ADMM`):** Certifies $\hat{v}_i = \nabla f_i(\hat{x}_i) + \mathcal{A}_i^*(\hat{y}) + \frac{1}{\tau_i}(x_i - \hat{x}_i) - g_i$ per block.
- **Spectral (`refine_icg`):** Refines in 1D singular-value coordinates and lifts via $P \operatorname{diag}(\cdot) Q^\top$ using one gradient $\nabla f_1(\hat{Z})$.
- **Zero-overhead certificates in `APD`:** Inner `PF.ACG` steps already have this form on the rescaled subproblem, so `APD` extracts $\hat{v}$ at every inner step with zero extra oracle calls.
