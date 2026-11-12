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
| Spectral oracle `SpectralOracle.m` + `refine_ICG.m` | `proxax.oracle.spectral` (`SpectralOracle`, `SpectralProblem`, `refine_icg`), `proxax.prox.norms.NuclearNorm` | Curvature redistribution $\xi$, thin-SVD 1D vector reduction, and exact matrix subgradient refinement. |
| Refinement $(y) \mapsto (\hat{x}, \hat{v})$ | `proxax.certificates.stationarity.refine` | Exact $\hat{v} \in \nabla f(\hat{x}) + \partial h(\hat{x})$ via one adaptive composite gradient step. |
| Proximal subproblem $\phi(x) + \frac{1}{2\lambda_k}\|x - z_k\|^2$ | `proxax.oracle.composite.ProxRegularizedOracle`, `RescaledProxOracle` | Splits $1/\lambda_k = \alpha_{\text{smooth}} + \mu_{\text{prox}}$ (`AIPP`) or rescales $(\phi + \hat{m}\|\cdot - z_k\|^2)/(2\hat{m})$ (`APD`). |
| Inner `ACG`, `PF.ACG`, & `AdapAPG` | `proxax.solvers.inner.acg`, `proxax.solvers.inner.pf_acg`, `proxax.solvers.unconstrained.adap_apg.AdapAPG` | Tracks $(u_t, \eta_t)$ (`ACG`), exact refinement-free $r_{j+1}$ (`PF.ACG`), and Lin & Xiao (2014) adaptive APG (`AdapAPG`). |
| Unconstrained `AIPP` & `APD` (`PF.APD`) | `proxax.solvers.unconstrained` (`AIPP`, `APD`) | Known-$m$ / adaptive-$\lambda_k$ `AIPP` and parameter-free `PF.APD` + Eq. (6.1) decay. |
| Direct composite & FISTA-type solvers (`ECG`, `AG`, `UPFAG`, `NC_FISTA`, `ADAP_FISTA`, `AC_ACG`) | `proxax.solvers.unconstrained` (`ECG`, `AG`, `UPFAG`, `NCFISTA`, `AdapFISTA`, `ACACG`) | Exact composite gradient, Ghadimi–Lan `AG` & `UPFAG`, Liang–Monteiro–Sim `NC-FISTA` / `ADAP-NC-FISTA`, and Liang–Monteiro `AC-ACG`. |
| Spectral ICG solvers (`IA_ICG.m`, `DA_ICG.m`) | `proxax.solvers.unconstrained` (`IAICG`, `DAICG`) | Inner-accelerated and doubly-accelerated spectral composite solvers (Kong & Monteiro 2020). |
| Set/cone constraint $\mathcal{A}(x) \in \mathcal{C}$ & Moreau envelope | `proxax.constraints.sets`, `proxax.constraints.linear_op`, `proxax.oracle.composite.PenalizedOracle` | Matrix or PyTree callable $\mathcal{A}$ (`jax.linear_transpose` adjoint) + any `ConvexSet` projection $\Pi_{\mathcal{C}}$. |
| Constrained penalty & AL frameworks (`penalty.m`, `iALM.m`, `AIDAL.m`, `sProxALM.m`, `IAIPAL.m`) | `proxax.solvers.constrained` (`QPAIPP`, `QPAPD`, `IALM`, `SProxALM`, `IAIPAL`) | Geometric penalty (`QPAIPP`, `QPAPD`), dampened proximal AL (`IALM`), smoothed proximal ALM (`SProxALM`), and single-`ACG`-step proximal AL (`IAIPAL`). |
| Optax step-by-step transforms | `proxax.optax_compat.transforms` (`proximal`, `aipp`, `apd`, `qp_aipp`, `qp_apd`, `ialm`) | Single-inner-step PyTree state machines compatible with `jax.jit` and `jax.lax.while_loop`. |

---

## 4. Completed `wwkong/nc_opt` Coverage & Extensions

1. **Complete `nc_opt` Solver & Framework Coverage:** Every solver and outer framework in `wwkong/nc_opt` (`ACG`, `PF.ACG`, `AIPP`, `APD`, `ECG`, `AG`, `UPFAG`, `NCFISTA`, `AdapFISTA`, `ACACG`, `AdapAPG`, `SpectralOracle` + `IAICG` + `DAICG`, `QPAIPP`, `QPAPD`, `IALM` / `NL-IAPIAL` / `AIDAL`, `SProxALM`, and `IAIPAL`) is implemented with exact KKT/subdifferential stationarity certificates and `jax.jit` compatibility.
2. **Future Extensions:** Add `sets.PSDCone` (`jnp.linalg.eigh`) and a standalone `benchmarks/` suite reproducing the paper tables from `wwkong/nc_opt/tests/papers/`.
