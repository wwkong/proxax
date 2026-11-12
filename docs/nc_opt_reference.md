# Reference & Roadmap from `wwkong/nc_opt`

`proxax` brings the certified nonconvex composite proximal-point methodology of **[`wwkong/nc_opt`](https://github.com/wwkong/nc_opt)** (MATLAB) into an idiomatic, `jit`/`vmap`-compatible **JAX + Optax** architecture.

---

## 1. Pointers to `wwkong/nc_opt`

- **Upstream Repository:** https://github.com/wwkong/nc_opt
- **Key Upstream Directories in `nc_opt`:**
  - `src/alg_framework/` — Core framework abstractions for outer proximal-point methods, adaptive line-search / curvature estimation, inner accelerated composite gradient (`ACG` / `AICG`) subroutines, and refinement maps.
  - `src/solvers/` — Concrete unconstrained and linearly/cone-constrained nonconvex composite solvers (`AIPP`, `R-AIPP`, `QP-AIPP`, `NL-IAPIAL` / dampened `iALM`, `AICG`, `AG`).
  - `src/prox/` & `tests/` — Nonsmooth regularizer / indicator proximal operators and canonical nonconvex test problems (nonconvex QP, spectral/matrix problems, sparse/box-constrained models).

### Primary Algorithmic References (Kong, Melo, Monteiro)

1. **Unconstrained `AIPP` & Adaptive `R-AIPP`:**
   - W. Kong, J. G. Melo, and R. D. C. Monteiro, *"Complexity of a quadratic penalty accelerated inexact proximal point method for solving linearly constrained nonconvex composite programs,"* SIAM Journal on Optimization, 29(4):2566–2593, 2019. (Establishes `AIPP` with $\sigma$-relative proximal error criterion and `QP-AIPP`.)
   - R. D. C. Monteiro and C. Ortiz / W. Kong et al. on **Relaxed / Adaptive AIPP (`R-AIPP`)**: dynamically adapting proximal step size $\lambda_k$ and local smoothness $L_k$ without requiring *a priori* knowledge of $(m, L^+)$, rejecting steps when either inner convexity invariants or outer proximal descent fail.
2. **Constrained Penalty & Dampened Inexact ALM (`QP-AIPP`, `NL-IAPIAL`):**
   - W. Kong, J. G. Melo, and R. D. C. Monteiro, *"Iteration-complexity of a proximal augmented Lagrangian method for solving nonconvex composite optimization problems with nonlinear convex constraints,"* Mathematics of Operations Research, 48(2), 2023 (`NL-IAPIAL` — dampened multiplier update $\bar{p}_k = (1 - \theta) p_{k-1}$ for general cone/set constraints).
   - W. Kong and R. D. C. Monteiro, *"An accelerated inexact dampened augmented Lagrangian method for linearly-constrained nonconvex composite optimization problems,"* Computational Optimization and Applications, 2023 (`AIDAL` / dampened `iALM` with constant/gentle penalty schedule).

---

## 2. Mapping `nc_opt` Concepts to `proxax` Modules

| `nc_opt` Concept / Component | `proxax` Module & Symbols | Notes |
| :--- | :--- | :--- |
| Oracle `(f, grad_f, h, prox_h)` + curvature `(m, L)` | `proxax.oracle.terms.SmoothTerm`, `proxax.prox.base.ProxOperator`, `proxax.oracle.composite.CompositeOracle` | Supports arbitrary JAX PyTrees and automatic `domain_project` onto $\operatorname{cl}(\operatorname{dom}(h))$. |
| Refinement $(y) \mapsto (\hat{x}, \hat{v})$ | `proxax.certificates.stationarity.refine` | Computes exact $\hat{v} \in \nabla f(\hat{x}) + \partial h(\hat{x})$ via one adaptive composite gradient step. |
| Proximal subproblem $\psi_k(x) = \phi(x) + \frac{1}{2\lambda_k}\|x - z_k\|^2$ | `proxax.oracle.composite.ProxRegularizedOracle` | Splits $1/\lambda_k = \alpha_{\text{smooth}} + \mu_{\text{prox}}$ so inner `ACG` exploits exact $\mu_{\text{prox}}$-strong convexity. |
| Inner solver (`ACG` + $\sigma$-criterion + $\varepsilon$-subgradient $(u_t, \eta_t)$) | `proxax.solvers.inner.acg` (`init_acg`, `step_acg`, `run_acg`) | Tracks $(u_t, \eta_t) \in \partial_{\eta_t} \psi_k(y_t)$ and checks convexity invariant when `check_convexity=True`. |
| Unconstrained `AIPP` & `R-AIPP` | `proxax.solvers.unconstrained.aipp.AIPP` | Unified solver handling both known-$m$ (`lower_curvature` provided) and adaptive-$\lambda_k$ (`R-AIPP`) modes. |
| Set/cone constraint $\mathcal{A}(x) \in \mathcal{C}$ & Moreau envelope | `proxax.constraints.sets`, `proxax.constraints.linear_op`, `proxax.oracle.composite.PenalizedOracle` | Matrix or PyTree callable $\mathcal{A}$ (`jax.linear_transpose` adjoint) + any `ConvexSet` projection $\Pi_{\mathcal{C}}$. |
| Constrained `QP-AIPP` | `proxax.solvers.constrained.qp_aipp.QPAIPP` | Geometric penalty continuation + `AIPP` carrying unpenalized $\lambda_k$ across penalty stages. |
| Optax step-by-step optimizer integration | `proxax.optax_compat.transforms` (`proximal`, `aipp`, `qp_aipp`) | Flattens inner `ACG` + outer `AIPP` / penalty transitions into PyTree state machines. |

---

## 3. Roadmap for Next Development Phases in WSL2

1. **Dampened Inexact Augmented Lagrangian (`iALM` / `AIDAL` / `NL-IAPIAL`) in `src/proxax/solvers/constrained/`:**
   - `PenalizedOracle` in `src/proxax/oracle/composite.py` **already implements** the exact dampened Moreau-envelope AL smooth term (`multiplier`, `penalty`, `dampen`) and `effective_dual` $\hat{p}(x) = c(w(x) - \Pi_{\mathcal{C}}(w(x))) \in \mathcal{N}_{\mathcal{C}}(\Pi_{\mathcal{C}}(w(x)))$!
   - Implement `IALM` (with dampened dual update $p_k = (1 - \theta) p_{k-1} + \chi_k (\dots)$ and conditional penalty growth test from `AIDAL` / `NL-IAPIAL`), wire `proxax.solve(..., method="ialm")`, and optionally expose `optax_compat.ialm`.
2. **Additional Unconstrained Baselines (`AG` / `AICG`) in `src/proxax/solvers/unconstrained/`:**
   - Direct nonconvex accelerated gradient (`AG`) and inexact composite gradient (`AICG`) baselines from `nc_opt` for head-to-head iteration/oracle complexity comparisons.
3. **Additional Proximal & Cone Primitives:**
   - Nuclear norm / singular-value soft-thresholding (`prox.NuclearNorm`), $\ell_0$ / capped-$\ell_1$ / SCAD / MCP composite decompositions, and positive semidefinite cone projection (`sets.PSDCone` via `jnp.linalg.eigh`).
4. **Separate `benchmarks/` Suite (Outside `tests/`):**
   - Keep `tests/` fast (< 10s) and place larger `nc_opt` parity benchmarks (spectral-norm QPs, sparse PCA, linearly/cone-constrained nonconvex QPs) in a dedicated `benchmarks/` directory.
