# Proxax Roadmap

This document tracks planned features, algorithmic extensions, and engineering
milestones for **Proxax** beyond the current `v0.0.5` baseline.

---

## Current Baseline (`v0.0.5`)

- **Problem classes (least to most specialized):**
  1. Unconstrained composite optimization: $\min_{x \in \mathbb{R}^n} f(x) + h(x)$
     (`px.CompositeProblem` — 9 solvers: `APD`, `AIPP`, `ACACG`, `AdapFISTA`,
     `NCFISTA`, `AdapAPG`, `UPFAG`, `AG`, `ECG`).
  2. Set- and cone-constrained composite optimization:
     $\min_{x \in \mathbb{R}^n} f(x) + h(x)$ s.t. $\mathcal{A}(x) \in \mathcal{C}$
     (`px.CompositeProblem` — 5 solvers: `IALM`, `QPAPD`, `QPAIPP`, `IAIPAL`,
     `SProxALM`).
  3. Min-max & saddle-point optimization:
     $\min_{x \in \mathcal{X}} \max_{y \in \mathcal{Y}} \Phi(x, y) + h(x) - g(y)$
     (`px.MinMaxProblem` — 3 solvers: `ExtraGradient`, `PDHG`,
     `ProxPointMinMax`).
  4. Matrix spectral composite optimization:
     $\min_{X \in \mathbb{R}^{p \times q}} f_1(X) + (f_2^{\mathcal{V}} \circ \sigma)(X) + (h^{\mathcal{V}} \circ \sigma)(X)$
     (`px.SpectralProblem` — 2 single-SVD solvers: `DAICG`, `IAICG`).
- **Core infrastructure:** Unified `from proxax import px` facade, PyTree-native
  oracles and composable state trackers, exact subdifferential/KKT refinement
  certificates, implicit differentiation (`px.implicit_solve`), single-step Optax
  transformations (`px.optax_compat`), `jax.jit` / `jax.vmap` support,
  reproducible `benchmarks/` suite, GitHub Actions CI (`.github/workflows/ci.yml`),
  Sphinx Book Theme documentation on Read the Docs, and automated PyPI
  publishing (`.github/workflows/publish.yml`).

---

## Milestone 1: Composite Min-Max & Saddle-Point Optimization (`px.MinMaxProblem`)

Extend Proxax to certified primal-dual saddle-point problems (slotting between
Constrained and Spectral in the problem hierarchy):

$$\min_{x \in \mathcal{X}} \max_{y \in \mathcal{Y}} \; \Phi(x, y) + h(x) - g(y)$$

- **Problem & certificate abstraction:**
  - `px.MinMaxProblem(coupling=..., prox_x=..., prox_y=...)` over arbitrary JAX
    PyTrees `(x, y)` with optional partial smoothness and weak-convexity /
    weak-concavity metadata $(m_x, m_y, L_{xx}, L_{xy}, L_{yy})$.
  - Exact primal-dual stationarity certificates
    $\hat{v}_x \in \nabla_x \Phi(\hat{x}, \hat{y}) + \partial h(\hat{x})$ and
    $\hat{v}_y \in -\nabla_y \Phi(\hat{x}, \hat{y}) + \partial g(\hat{y})$
    (plus primal-dual gap certificates in the convex-concave regime).
- **Candidate solvers (`px.solvers`):**
  - *Convex-concave & bilinear:* Primal-Dual Hybrid Gradient (`PDHG` /
    Chambolle–Pock) and Optimistic / Extra-Gradient (`OGDA`, `ExtraGradient`).
  - *Nonconvex-concave & weakly-convex–weakly-concave:* Inexact proximal-point
    and Moreau-envelope saddle-point methods built on `ACG` and parameter-free
    `PF.ACG` subproblem solvers, plus single-loop smoothed GDA.
- **Optax integration:** Single-step primal-dual state machines in
  `px.optax_compat`.

---

## Milestone 2: Reproducible Benchmark Suite (`benchmarks/`)

Build a self-contained benchmarking and plotting suite (kept separate from the
fast unit tests in `tests/` and isolated under an optional `benchmarks`
dependency group in `pyproject.toml`):

- **Standardized problem suite:**
  - Indefinite box- and simplex-constrained QPs across conditioning and
    negative-curvature ratios $m / M$.
  - Sparse regression and nonconvex margin classification (`L1`, `ElasticNet`,
    MCP, `tanh` loss).
  - Linearly- and cone-constrained problems ($k$-Fantope Sparse PCA, portfolio
    SOCPs, bounded matrix completion).
  - Low-rank spectral matrix completion (convex nuclear norm vs. nonconvex
    log-sum / MCP singular-value penalties).
- **Metrics & baselines:**
  - Certified stationarity / KKT residual vs. oracle evaluations and wall-clock
    time on CPU and GPU (`float32` and `float64`).
  - Batched accelerator throughput under `jax.jit(jax.vmap(...))` for batch
    sizes $B \in \{1, 16, 256, 4096\}$.
  - External comparisons against `optax` (L-BFGS, Adam, proximal SGD),
    `optimistix` / `jaxopt`, and `osqp` / `scs`, generating Dolan–Moré
    performance profiles and summary tables for the documentation.

---

## Milestone 3: Implicit Differentiation Through `px.solve`

Enable end-to-end differentiation $\partial \hat{x}(\theta) / \partial \theta$
through `px.solve` via `jax.custom_vjp` without unrolling inner
`jax.lax.while_loop` iterations:

- **Unconstrained composite fixed-point implicit VJP:** Differentiate
  $\hat{x} = \operatorname{prox}_{\tau h}(\hat{x} - \tau \nabla_x f(\hat{x}, \theta))$
  using matrix-free `jax.scipy.sparse.linalg.cg` / `gmres` and JVPs of
  `prox_op.prox`.
- **Constrained KKT implicit VJP:** Differentiate primal-dual KKT solutions
  $(\hat{x}(\theta), \hat{p}(\theta))$ returned by `IALM` and `QPAPD` for
  bilevel optimization, hyperparameter learning, and differentiable neural
  layers.

---

## Milestone 4: Folded-Concave Regularizers & Cone/Prox Primitives

Reduce boilerplate for common structured penalties and matrix/vector constraints:

- **Nonconvex regularizer decompositions:** Turnkey **SCAD**, **MCP**,
  **Log-Sum**, and **Capped-$\ell_1$** helpers that automatically split a
  folded-concave penalty $p_\lambda(x) = q_\lambda(x) + \lambda \|x\|_1$ into
  its smooth $(m, M)$-weakly convex part (`SmoothTerm`) and convex `px.prox.L1`
  part.
- **Additional `ProxOperator` & `ConvexSet` primitives:**
  - Symmetric positive semidefinite cone (`px.constraints.PSDSet`) and built-in
    $k$-Fantope (`px.prox.Fantope`).
  - Group Lasso (`L21`), $\ell_\infty$ ball/norm, and 1D total variation.

---

## Milestone 5: Block-Constrained Composite Optimization & Splitting (`ADMM`)

Add dedicated support for separable multi-block and consensus composite
problems:

$$\min_{x_1, \dots, x_K} \; \sum_{i=1}^K \big(f_i(x_i) + h_i(x_i)\big) \quad \text{s.t.} \quad \sum_{i=1}^K \mathcal{A}_i(x_i) = b \;\; (\text{or } \in \mathcal{C})$$

- **Block problem & oracle abstraction:** Separable block terms
  $(f_i, h_i, \mathcal{A}_i)$ over arbitrary PyTree blocks with certified KKT
  primal/dual stationarity and feasibility residuals.
- **Splitting solvers (`px.solvers`):** Certified proximal `ADMM` for two-block,
  multi-block, and consensus composite optimization (covering both convex and
  weakly-convex smooth blocks).

---

## Milestone 6: Example Smoke Checks & `v0.1.0` Release

- **CI coverage (`.github/workflows/ci.yml`):** Add Doctest / runnable example
  smoke checks alongside the existing `ruff`, `mypy`, `pytest` (`float32` &
  `float64`), and `sphinx-build` gates.
- **API & example polish:** Standardize example formatting and tag `v0.1.0`.
