# Reference & Roadmap from `wwkong/nc_opt` and Kong's `APD`

`proxax` brings the certified nonconvex composite proximal-point methodology of **[`wwkong/nc_opt`](https://github.com/wwkong/nc_opt)** (MATLAB) into an idiomatic, `jit`/`vmap`-compatible **JAX + Optax** architecture.

---

## 1. Pointers to `wwkong/nc_opt` & Primary Papers

- **Upstream MATLAB Repository:** https://github.com/wwkong/nc_opt
  - `src/alg_framework/` — Core outer proximal-point templates, adaptive curvature / line-search routines, inner `ACG` subroutines, and refinement maps.
  - `src/solvers/` — Unconstrained and linearly/cone-constrained composite solvers (`AIPP`, `R-AIPP`, `APD` / `PF.APD`, `QP-AIPP`, `NL-IAPIAL`, `AIDAL`, `AG`).
  - `tests/papers/apd/` — Reference MATLAB implementation and experiments for `PF.APD` (`APD`): https://github.com/wwkong/nc_opt/tree/master/tests/papers/apd
- **Author Publications Page:** https://wwkong.github.io/publication/

### Key Papers by Weiwei (William) Kong et al.

1. **Parameter-Free Accelerated Proximal Descent (`PF.APD` / `APD`):**
   - W. Kong, *"Complexity-Optimal and Parameter-Free First-Order Methods for Finding Stationary Points of Composite Optimization Problems,"* SIAM Journal on Optimization, 34(3):3005–3032, 2024 (DOI: `10.1137/22M1498826`, PDF: https://wwkong.github.io/files/publications/apd.pdf).
2. **Unconstrained `AIPP` & Constrained `QP-AIPP`:**
   - W. Kong, J. G. Melo, and R. D. C. Monteiro, *"Complexity of a quadratic penalty accelerated inexact proximal point method for solving linearly constrained nonconvex composite programs,"* SIAM Journal on Optimization, 29(4):2566–2593, 2019.
   - W. Kong, J. G. Melo, and R. D. C. Monteiro, *"An efficient adaptive accelerated inexact proximal point method for solving linearly constrained nonconvex composite problems,"* Comput. Math. Appl., 76:305–346, 2020 (`R-QP-AIPP`).
3. **Proximal & Dampened Augmented Lagrangian (`NL-IAPIAL` / `AIDAL`):**
   - W. Kong, J. G. Melo, and R. D. C. Monteiro, *"Iteration-complexity of a proximal augmented Lagrangian method for solving nonconvex composite optimization problems with nonlinear convex constraints,"* Math. Oper. Res., 48(2):1066–1094, 2023.
   - W. Kong, J. G. Melo, and R. D. C. Monteiro, *"Iteration-complexity of an inner accelerated inexact proximal augmented Lagrangian method based on the classical Lagrangian function,"* SIAM Journal on Optimization, 33(1):181–210, 2023.

---

## 2. Deep Dive: Kong's `APD` (`PF.APD` + `PF.ACG`, SIAM J. Optim. 2024)

Unlike "Accelerated Primal-Dual" methods in saddle-point literature, Kong's **`APD`** stands for **Parameter-Free Accelerated Proximal Descent (`PF.APD`)** (https://wwkong.github.io/files/publications/apd.pdf, Algorithms 3.1–3.4). It is a premier target for `proxax`'s next unconstrained and penalty-subproblem solver:

- **Truly Parameter-Free & Optimal in Both Regimes:** Requires **no knowledge** of $(m, M)$ or boundedness of $\operatorname{dom} h$, achieving $\tilde{\mathcal{O}}(\sqrt{m M}\Delta_0 / \varepsilon^2)$ resolvent evaluations when $f$ is $m$-weakly convex and automatically improving to $\tilde{\mathcal{O}}(\sqrt{M}\min\{\sqrt{\Delta_0}/\varepsilon, d_0/\sqrt{\varepsilon}\})$ when $f$ is convex.
- **Rescaled Subproblem (`Algorithm 3.4`, Eq. (3.12)):** At outer step $k$ with lower-curvature estimate $\hat{m} > 0$, `PF.ACG` (`Algorithm 3.2`) is called with $\mu = 1/2$ on:

  $$\psi^s(z) = \frac{f(z)}{2\hat{m}} + \frac{1}{2}\|z - z_k\|^2, \qquad \psi^n(z) = \frac{h(z)}{2\hat{m}}.$$

- **Exact Certificate at Every Inner Iteration (No Extra `refine` Needed):**
  In `Algorithm 3.1`, $y_{j+1} = \operatorname{prox}_{\psi^n / (L_{j+1} + \mu)}\big(\tilde{x}_j - \nabla \psi^s(\tilde{x}_j) / (L_{j+1} + \mu)\big)$ with line search $\psi^s(y_{j+1}) - \ell_{\psi^s}(y_{j+1}; \tilde{x}_j) \le \frac{L_{j+1}}{2}\|y_{j+1} - \tilde{x}_j\|^2$, and `Algorithm 3.2` forms:

  $$r_{j+1} = \nabla \psi^s(y_{j+1}) - \nabla \psi^s(\tilde{x}_j) + (L_{j+1} + \mu)(\tilde{x}_j - y_{j+1}) \in \nabla \psi^s(y_{j+1}) + \partial \psi^n(y_{j+1}).$$

  Rescaling gives $v = 2\hat{m}(r_{j+1} + z_k - y_{j+1}) \in \nabla f(y_{j+1}) + \partial h(y_{j+1})$ directly!
- **Dynamic "Bad" vs. "Good" Inner Stopping Checks (Eqs. (3.5)–(3.6)):**
  - **Bad termination (`Step 2`, Eq. (3.5)):** If $\mu A_{j+1}\|y_{j+1} - \tilde{x}_j\|^2 \le \|y_{j+1} - y_0\|^2$ or $\psi(y_0) \ge \psi(y_{j+1}) + \langle r_{j+1}, y_0 - y_{j+1}\rangle$ fails, local $\mu$-strong convexity is violated; `Algorithm 3.3` multiplies $\hat{m} \leftarrow \alpha \hat{m}$ and retries.
  - **Good termination (`Step 3`, Eq. (3.6)):** Stops when $\|r_{j+1}\|^2 \le \sigma^2 \|y_{j+1} - y_0\|^2$ and $\|r_{j+1} + y_0 - y_{j+1}\|^2 \le \theta\big[\psi(y_0) - \psi(y_{j+1}) + \frac{1}{2}\|y_{j+1} - y_0\|^2\big]$ ($\theta > 2$, $\sigma = \rho \in (0, 1)$), certifying proximal descent (Eqs. (2.2)–(2.4)).
- **Practical Curvature Decay Heuristic (Eq. (6.1)):**
  In `nc_opt/tests/papers/apd`, $\alpha = \beta = 2$, $\rho = 1/\sqrt{\alpha}$, with $L_0 \leftarrow L_0 / (1 + \beta/2)$ before `PF.ACG` and $m_{k+1} \leftarrow \max\{m_0, m_{k+1} / (1 + \alpha/2)\}$ across outer steps (guarding against the `float32` noise floor in JAX).
- **Constrained Penalty / ALM Integration (Sections 5 & 7 of `apd.pdf`):**
  Using `APD` as the inner stage solver inside `QPAIPP` (`QP-APD`, with $m_0 = \varepsilon$) or proximal `iALM` yields a **completely parameter-free** constrained solver with $\tilde{\mathcal{O}}(\varepsilon^{-3})$ nonconvex complexity that automatically improves to $\tilde{\mathcal{O}}(\varepsilon^{-1.5})$ when $f$ is convex!

---

## 3. Mapping Current `proxax` Modules to `nc_opt`

| `nc_opt` Concept / Component | `proxax` Module & Symbols | Notes |
| :--- | :--- | :--- |
| Oracle `(f, grad_f, h, prox_h)` + curvature `(m, M)` | `proxax.oracle.terms.SmoothTerm`, `proxax.prox.base.ProxOperator`, `proxax.oracle.composite.CompositeOracle` | Arbitrary JAX PyTrees + automatic `domain_project` onto $\operatorname{cl}(\operatorname{dom}(h))$. |
| Refinement $(y) \mapsto (\hat{x}, \hat{v})$ | `proxax.certificates.stationarity.refine` | Exact $\hat{v} \in \nabla f(\hat{x}) + \partial h(\hat{x})$ via one adaptive composite gradient step. |
| Proximal subproblem $\phi(x) + \frac{1}{2\lambda_k}\|x - z_k\|^2$ | `proxax.oracle.composite.ProxRegularizedOracle` | Splits $1/\lambda_k = \alpha_{\text{smooth}} + \mu_{\text{prox}}$ so inner `ACG` exploits $\mu_{\text{prox}}$-strong convexity. |
| Inner `ACG` ($\sigma$-criterion + $(u_t, \eta_t)$) | `proxax.solvers.inner.acg` (`init_acg`, `step_acg`, `run_acg`) | Tracks $(u_t, \eta_t) \in \partial_{\eta_t} \psi_k(y_t)$ and convexity check. |
| Unconstrained `AIPP` & adaptive `R-AIPP` | `proxax.solvers.unconstrained.aipp.AIPP` | Supports both known-$m$ (`lower_curvature`) and adaptive-$\lambda_k$ modes. |
| Set/cone constraint $\mathcal{A}(x) \in \mathcal{C}$ & Moreau envelope | `proxax.constraints.sets`, `proxax.constraints.linear_op`, `proxax.oracle.composite.PenalizedOracle` | Matrix or PyTree callable $\mathcal{A}$ (`jax.linear_transpose` adjoint) + any `ConvexSet` projection $\Pi_{\mathcal{C}}$. |
| Constrained `QP-AIPP` | `proxax.solvers.constrained.qp_aipp.QPAIPP` | Geometric penalty continuation + `AIPP` preserving unpenalized $\lambda_k$. |
| Optax step-by-step transforms | `proxax.optax_compat.transforms` (`proximal`, `aipp`, `qp_aipp`) | Single-inner-step PyTree state machines compatible with `jax.jit` and `jax.lax.while_loop`. |

---

## 4. Prioritized Roadmap for WSL2 Development

1. **Implement `APD` (`PF.APD` + `PF.ACG`) in `src/proxax/solvers/unconstrained/apd.py` (and `solvers/inner/pf_acg.py`):**
   - Implement Algorithms 3.1–3.4 + optional practical decay Eq. (6.1) from https://wwkong.github.io/files/publications/apd.pdf, wire `proxax.solve(..., method="apd")`, and allow `QPAIPP` (and `optax_compat`) to use `APD` as a parameter-free subproblem solver (`QP-APD`).
2. **Dampened Inexact Augmented Lagrangian (`iALM` / `NL-IAPIAL` / `AIDAL`) in `src/proxax/solvers/constrained/`:**
   - `PenalizedOracle` in `src/proxax/oracle/composite.py` already implements the dampened Moreau-envelope AL smooth term (`multiplier`, `penalty`, `dampen`) and `effective_dual` $\hat{p}(x)$. Pair it with `AIPP` and `APD` (`PF.APD`-powered parameter-free `iALM`, as proposed in Section 7 of `apd.pdf`!).
3. **Additional Matrix/Cone Primitives & `benchmarks/` Suite:**
   - Add `prox.NuclearNorm` (SVD soft-thresholding, used in LRMC Eq. (6.4) of `apd.pdf`) and `sets.PSDCone` (`jnp.linalg.eigh`, used in QSDP Eq. (6.2) of `apd.pdf`), plus a standalone `benchmarks/` folder (keeping `tests/` tiny and `< 10s`).
