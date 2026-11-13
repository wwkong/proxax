# Appendix A: Literature Review & Algorithm Deep Dives

This appendix summarizes the first-order optimization literature, worst-case
oracle complexity bounds, and algorithmic mechanics behind the 16 solvers in
`proxax`.

---

## 1. Complexity & Literature Overview Across Solvers

Below, $\rho > 0$ denotes the stationarity tolerance $\|\hat{v}\| \le \rho$,
$\eta > 0$ denotes the feasibility tolerance
$\|\mathcal{A}(\hat{x}) - \Pi_{\mathcal{C}}(\mathcal{A}(\hat{x}))\| \le \eta$,
$\Delta_0 := \phi(x_0) - \phi_\star$ is the initial optimality gap, and
$(m, M)$ are the lower (weak-convexity) and upper (smoothness) curvature
constants of $f$.

### Unconstrained Composite Solvers ($\min_x f(x) + h(x)$)

| `proxax` Class | Algorithm Name | Curvature Input | Nonconvex Complexity ($m > 0$) | Convex Regime ($m = 0$) | Primary Reference |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `APD` | `PF.APD` (`PF.ACG` inner) | **None** | $\tilde{\mathcal{O}}\!\left(\frac{\sqrt{m M}\,\Delta_0}{\rho^2}\right)$ | $\tilde{\mathcal{O}}\!\left(\sqrt{M}\min\!\left\{\frac{\sqrt{\Delta_0}}{\rho}, \frac{d_0}{\sqrt{\rho}}\right\}\right)$ | Kong (2024) |
| `AIPP` | `R-AIPP` (`ACG` inner) | Optional $m$ | $\tilde{\mathcal{O}}\!\left(\frac{\sqrt{m M}\,\Delta_0}{\rho^2}\right)$ | $\tilde{\mathcal{O}}\!\left(\frac{\sqrt{M}\,\Delta_0}{\rho^2}\right)$ (adaptive $\lambda_k$) | Kong, Melo & Monteiro (2019, 2020) |
| `ACACG` | `AC-ACG` | Optional $M$ | Adaptive average-curvature acceleration | Accelerated $\mathcal{O}(\sqrt{M} / \rho)$ | Liang & Monteiro (2021) |
| `AdapFISTA` | `ADAP-NC-FISTA` | **None** | $\tilde{\mathcal{O}}\!\left(\frac{\sqrt{m M}\,\Delta_0}{\rho^2}\right)$ | Accelerated FISTA rate | Liang, Monteiro & Sim (2022) |
| `NCFISTA` | `NC-FISTA` | $(m, M)$ or $M$ | $\mathcal{O}\!\left(\frac{\sqrt{m M}\,\Delta_0}{\rho^2}\right)$ | Accelerated FISTA rate | Liang, Monteiro & Sim (2022) |
| `AdapAPG` | Adaptive APG | **None** (opt. $\mu$) | Adaptive backtracking / relaxation | Optimal $\mathcal{O}(\sqrt{M/\mu}\log(1/\rho))$ or $\mathcal{O}(\sqrt{M}/\rho)$ | Lin & Xiao (2014) |
| `UPFAG` | Unified Parameter-Free AG | **None** | $\mathcal{O}\!\left(\frac{M \Delta_0}{\rho^2}\right)$ | $\mathcal{O}\!\left(\frac{\sqrt{M} d_0}{\sqrt{\rho}}\right)$ | Lan, Ouyang & Zhang (2023) |
| `AG` | Nonconvex AG | Optional $M$ | $\mathcal{O}\!\left(\frac{M \Delta_0}{\rho^2}\right)$ | $\mathcal{O}\!\left(\frac{\sqrt{M} d_0}{\sqrt{\rho}}\right)$ | Ghadimi & Lan (2016) |
| `ECG` | Composite Backtracking PG | **None** | $\mathcal{O}\!\left(\frac{M \Delta_0}{\rho^2}\right)$ | $\mathcal{O}\!\left(\frac{M \Delta_0}{\rho^2}\right)$ | Beck (2017) |

### Matrix Spectral Solvers ($\min_X f_1(X) + (f_2^{\mathcal{V}} + h^{\mathcal{V}})(\sigma(X))$)

| `proxax` Class | Algorithm Name | Matrix SVDs / Outer Step | Inner Vector Solver | Primary Reference |
| :--- | :--- | :--- | :--- | :--- |
| `IAICG` | `IA-ICG` | **1 thin SVD** | 1D `ACG` on $\mathbb{R}^{\min(p, q)}$ | Kong & Monteiro (2020) |
| `DAICG` | `DA-ICG` | **1 thin SVD** (2 with monotone check) | Outer FISTA extrapolation + 1D `ACG` | Kong & Monteiro (2020) |

### Set- and Cone-Constrained Solvers ($\min_x f(x) + h(x)$ s.t. $\mathcal{A}(x) \in \mathcal{C}$)

| `proxax` Class | Algorithm Name | Curvature Input | Complexity ($\rho = \eta = \varepsilon$) | Primary Reference |
| :--- | :--- | :--- | :--- | :--- |
| `QPAPD` | `QP-APD` | **None** | $\tilde{\mathcal{O}}(\varepsilon^{-3})$ nonconvex; $\tilde{\mathcal{O}}(\varepsilon^{-1.5})$ convex | Kong (2024) |
| `QPAIPP` | `R-QP-AIPP` | Optional $m$ | $\tilde{\mathcal{O}}(\varepsilon^{-3})$ | Kong, Melo & Monteiro (2019, 2020) |
| `IALM` | `NL-IAPIAL` ($\theta=0$) / `AIDAL` ($\theta \in (0,1)$) | **None** (with `APD` inner) | $\tilde{\mathcal{O}}(\varepsilon^{-3})$ ($\tilde{\mathcal{O}}(\varepsilon^{-2.5})$ under regularity) | Kong, Melo & Monteiro (2023a) |
| `IAIPAL` | `IAIPAL` (1 `ACG` step / outer iter) | Optional $(m, M)$ | $\tilde{\mathcal{O}}(\varepsilon^{-3})$ | Kong, Melo & Monteiro (2023b) |
| `SProxALM` | `s-Prox-ALM` | Optional $(L, \|\mathcal{A}\|)$ | $\mathcal{O}(\varepsilon^{-2})$ on polyhedral/box constraints | Zhang & Luo (2020) |

---

## 2. Deep Dive: Parameter-Free `APD` (`PF.APD` + `PF.ACG`)

Unlike primal-dual saddle-point methods that share the acronym, Kong's **`APD`**
stands for **Parameter-Free Accelerated Proximal Descent (`PF.APD`)** (Kong,
*SIAM J. Optim.*, 2024, Algorithms 3.1–3.4):

1. **Rescaled Proximal Subproblem:** At outer stage $k$ with adaptive lower
   curvature estimate $\hat{m}_k > 0$, `APD` calls `PF.ACG` with strong
   convexity modulus $\mu = 1/2$ on the rescaled composite objective
   (`RescaledProxOracle`):

   $$\psi^s(z) := \frac{f(z)}{2\hat{m}_k} + \frac{1}{2}\|z - z_k\|_2^2, \qquad \psi^n(z) := \frac{h(z)}{2\hat{m}_k}.$$

2. **Refinement-Free Inner Certificates:** Each inner `PF.ACG` iteration
   computes $y_{j+1} = \operatorname{prox}_{\psi^n / (L_{j+1} + \mu)}\big(\tilde{x}_j - \nabla \psi^s(\tilde{x}_j)/(L_{j+1} + \mu)\big)$
   satisfying $\psi^s(y_{j+1}) - \ell_{\psi^s}(y_{j+1}; \tilde{x}_j) \le \frac{L_{j+1}}{2}\|y_{j+1} - \tilde{x}_j\|_2^2$,
   which immediately yields an exact subgradient of $\psi$:

   $$r_{j+1} := \nabla \psi^s(y_{j+1}) - \nabla \psi^s(\tilde{x}_j) + (L_{j+1} + \mu)(\tilde{x}_j - y_{j+1}) \in \nabla \psi^s(y_{j+1}) + \partial \psi^n(y_{j+1}).$$

   Unscaling gives $\hat{v} = 2\hat{m}_k(r_{j+1} + z_k - y_{j+1}) \in \nabla f(y_{j+1}) + \partial h(y_{j+1})$
   at every inner step with **zero extra proximal or gradient calls**.
3. **"Bad" vs. "Good" Termination Checks:**
   - **Bad termination (curvature violation):** If $\mu A_{j+1}\|y_{j+1} - \tilde{x}_j\|_2^2 \le \|y_{j+1} - y_0\|_2^2$
     or $\psi(y_0) < \psi(y_{j+1}) + \langle r_{j+1}, y_0 - y_{j+1} \rangle$, the
     assumed $\mu$-strong convexity of $\psi$ has been violated locally; `APD`
     multiplies $\hat{m}_k \leftarrow \alpha \hat{m}_k$ ($\alpha > 1$) and
     re-solves from $z_k$.
   - **Good termination (certified proximal descent):** When $\|r_{j+1}\|_2^2 \le \sigma^2 \|y_{j+1} - y_0\|_2^2$
     and $\|r_{j+1} + y_0 - y_{j+1}\|_2^2 \le \theta\big[\psi(y_0) - \psi(y_{j+1}) + \frac{1}{2}\|y_{j+1} - y_0\|_2^2\big]$,
     the stage accepts $z_{k+1} = y_{j+1}$ with guaranteed composite descent and
     relaxes $\hat{m}_{k+1} \leftarrow \max\{m_0, \hat{m}_k / (1 + \alpha/2)\}$
     so convex regions automatically enjoy larger proximal steps.

---

## 3. Deep Dive: Spectral Curvature Redistribution (`IA-ICG` & `DA-ICG`)

For $\Phi(X) = f_1(X) + (f_2^{\mathcal{V}} \circ \sigma)(X) + (h^{\mathcal{V}} \circ \sigma)(X)$
on $\mathbb{R}^{p \times q}$, applying a generic matrix proximal-point method
would evaluate $\operatorname{prox}_{\tau (h^{\mathcal{V}} \circ \sigma)}$ (a
full matrix SVD) on every inner `ACG` step and line-search trial.

Kong and Monteiro (2020) avoid inner matrix SVDs via two observations:

1. **Curvature Redistribution:** Because $\|X\|_F^2 = \|\sigma(X)\|_2^2$,
   setting $\xi = m_2$ shifts all weak convexity out of $f_2^{\mathcal{V}}$ into
   $f_1^\xi(X) = f_1(X) - \frac{\xi}{2}\|X\|_F^2$, rendering
   $f_2^{\mathcal{V},\xi}(s) = f_2^{\mathcal{V}}(s) + \frac{\xi}{2}\|s\|_2^2$
   **convex** on $\mathbb{R}^{\min(p, q)}$.
2. **Von Neumann Trace Reduction:** Linearizing $f_1^\xi$ at $Z_0$ gives a
   matrix subproblem whose linear + Frobenius term depends only on
   $\|X - (Z_0 - \lambda \nabla f_1^\xi(Z_0))\|_F^2$. Factoring
   $Z_0 - \lambda \nabla f_1^\xi(Z_0) = P \operatorname{diag}(x_0) Q^\top$ once
   via a thin SVD fixes the left and right singular vectors $(P, Q)$ for the
   entire inner `ACG` solve on $\mathbb{R}^{\min(p, q)}$. `DAICG` wraps this 1D
   inexact composite gradient step inside an outer Nesterov/FISTA extrapolation
   loop over $\mathbb{R}^{p \times q}$.

In `proxax`, SVD reconstructions $P \operatorname{diag}(s) Q^\top$ and diagonal
projections $\operatorname{diag}(P^\top M Q)$ explicitly enforce
`jax.lax.Precision.HIGHEST` so NVIDIA Tensor-Core TF32 truncation (`10`-bit
mantissa) cannot corrupt `float32` stationarity certificates.

---

## 4. Deep Dive: Penalty vs. Dampened Proximal ALM

For constrained problems $\min_x f(x) + h(x)$ subject to
$\mathcal{A}(x) \in \mathcal{C}$, `proxax` implements three complementary outer
mechanisms on top of the smooth Moreau-envelope penalty
$\psi_c(x; \bar{p}) = \frac{c}{2}\|w(x) - \Pi_{\mathcal{C}}(w(x))\|_2^2 - \frac{1}{2c}\|\bar{p}\|_2^2$,
where $w(x) = \mathcal{A}(x) + \bar{p}/c$:

1. **Pure Quadratic Penalty (`QPAPD`, `QPAIPP`):** Fixes $\bar{p} = 0$, solves
   $\min_x f(x) + \psi_{c_k}(x; 0) + h(x)$ to stationarity tolerance $\rho$, and
   increases $c_{k+1} = \gamma c_k$ until the feasibility residual satisfies
   $\|\mathcal{A}(\hat{x}_k) - \Pi_{\mathcal{C}}(\mathcal{A}(\hat{x}_k))\| \le \eta$.
   Because $\hat{p}_k = c_k(\mathcal{A}(\hat{x}_k) - \Pi_{\mathcal{C}}(\mathcal{A}(\hat{x}_k)))$
   still lies in $N_{\mathcal{C}}(\Pi_{\mathcal{C}}(\mathcal{A}(\hat{x}_k)))$,
   penalty methods produce an exact effective dual multiplier without storing
   dual iterates.
2. **Proximal & Dampened Augmented Lagrangian (`IALM` / `NL-IAPIAL` / `AIDAL`):**
   Maintains a multiplier $p_k \in \mathcal{Y}$, forms the dampened anchor
   $\bar{p}_k = (1 - \theta) p_k$ ($\theta \in [0, 1)$), solves the proximal AL
   subproblem via `APD` or `AIPP`, and updates

   $$p_{k+1} = \bar{p}_k + \chi c_k \big(w(\hat{x}_k) - \Pi_{\mathcal{C}}(w(\hat{x}_k))\big), \qquad \chi \in (0, 1].$$

   Setting $\theta = 0, \chi = 1$ recovers `NL-IAPIAL`; setting $\theta \in (0, 1)$
   (e.g., $\theta = 0.25, \chi = 0.5$ in `AIDAL`) contracts dual norm growth on
   highly nonconvex instances while allowing much smaller penalties $c_k$ than
   pure penalty methods.
3. **Single-Loop Smoothed Proximal ALM (`SProxALM`):** Maintains an exponentially
   smoothed primal anchor $z_{k+1} = z_k + \beta(x_{k+1} - z_k)$, performing
   only a single composite gradient step, dual ascent step, and anchor update per
   iteration without nested inner loops.

---

## 5. Bibliography

1. **Beck, A.** (2017). *First-Order Methods in Optimization*. MOS-SIAM Series
   on Optimization.
2. **Ghadimi, S., & Lan, G.** (2016). Accelerated gradient methods for
   nonconvex nonlinear and stochastic programming. *Mathematical Programming*,
   156(1–2), 59–99.
3. **Kong, W., Melo, J. G., & Monteiro, R. D. C.** (2019). Complexity of a
   quadratic penalty accelerated inexact proximal point method for solving
   linearly constrained nonconvex composite programs. *SIAM Journal on
   Optimization*, 29(4), 2566–2593.
4. **Kong, W., Melo, J. G., & Monteiro, R. D. C.** (2020). An efficient adaptive
   accelerated inexact proximal point method for solving linearly constrained
   nonconvex composite problems. *Computational Optimization and Applications*,
   76, 305–346.
5. **Kong, W., & Monteiro, R. D. C.** (2020). Accelerated inexact composite
   gradient methods for nonconvex spectral optimization problems.
   *arXiv:2007.11772* (*Computational Optimization and Applications*, 2023).
6. **Kong, W., Melo, J. G., & Monteiro, R. D. C.** (2023a). Iteration-complexity
   of a proximal augmented Lagrangian method for solving nonconvex composite
   optimization problems with nonlinear convex constraints. *Mathematics of
   Operations Research*, 48(2), 1066–1094.
7. **Kong, W., Melo, J. G., & Monteiro, R. D. C.** (2023b). Iteration-complexity
   of an inner accelerated inexact proximal augmented Lagrangian method based on
   the classical Lagrangian function. *SIAM Journal on Optimization*, 33(1),
   181–210.
8. **Kong, W.** (2024). Complexity-optimal and parameter-free first-order
   methods for finding stationary points of composite optimization problems.
   *SIAM Journal on Optimization*, 34(3), 3005–3032.
9. **Lan, G., Ouyang, Y., & Zhang, Z.** (2023). Unified and parameter-free
   accelerated gradient methods for convex and nonconvex optimization.
10. **Liang, J., & Monteiro, R. D. C.** (2021). An average curvature accelerated
    composite gradient method for nonconvex smooth composite optimization
    problems. *SIAM Journal on Optimization*, 31(1), 217–243.
11. **Liang, J., Monteiro, R. D. C., & Sim, C. K.** (2022). A FISTA-type
    accelerated gradient algorithm for solving smooth nonconvex composite
    optimization problems. *Computational Optimization and Applications*, 79,
    629–659.
12. **Lin, Q., & Xiao, L.** (2014). An adaptive accelerated proximal gradient
    method and its homotopy continuation for sparse optimization. *Computational
    Optimization and Applications*, 60(3), 633–674.
13. **Zhang, J., & Luo, Z.-Q.** (2020). A proximal alternating direction method
    of multipliers for linearly constrained nonconvex minimization / A global
    dual error bound and its application to the analysis of linearly constrained
    nonconvex optimization. *SIAM Journal on Optimization*, 30(3), 2272–2302.
