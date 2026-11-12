# Architecture & MVP Implementation Plan: `proxax`

## 1. Core Architectural Pillars & Standards

**`proxax`** is a JAX/Optax-native Python library for certified first-order proximal point methods on unconstrained and linearly/set-constrained nonconvex composite optimization problems.

### Engineering & Style Standards
- **Dependencies:** Strictly **`jax`** and **`optax`** at runtime (leveraging `optax.projections` and `optax.tree` directly; zero third-party framework lock-in).
- **Code Style:** **Google Python Style Guide**, strictly enforced via `ruff` (`convention = "google"`, module-only imports via `flake8-tidy-imports` / `isort`) and `pyright`:
  - Module-only imports everywhere (`import dataclasses`, `import typing`, `from collections import abc`, `from proxax import prox`).
  - `@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)` for all configs, solvers, problems, states, and results (paired with `@jax.tree_util.register_dataclass` for PyTree states/oracles). Unary math primitives (e.g., `prox.L1(0.1)`) keep their single primary value positional and optional metadata `kw_only`.
  - Structural subtyping via **`typing.Protocol`** (`ProxOperator`, `ConvexSet`, `LinearOperator`, `InnerCriterion`).
- **Documentation:** **MkDocs Material + `mkdocstrings[python]` + KaTeX** (Google-style docstrings + cross-linked `jax`/`optax` inventories).
- **Version Control:** Git (`main` branch initialized in repo root, atomic commits per milestone).

---

## 2. Unified Mathematical Formulations

### 2.1. Unconstrained Composite Problem
$$
\min_{x \in \mathcal{X}} \; \phi(x) := f(x) + h(x)
$$
- $\mathcal{X}$: Arbitrary PyTree of JAX arrays (inner product $\langle u, v \rangle = \text{optax.tree.vdot}(u, v)$).
- $f: \mathcal{X} \to \mathbb{R}$ (`smooth`): Differentiable, $(m, L^+)$-weakly convex/smooth ($-\frac{m}{2}\|x-y\|^2 \le f(x) - f(y) - \langle \nabla f(y), x-y \rangle \le \frac{L^+}{2}\|x-y\|^2$).
- $h: \mathcal{X} \to (-\infty, +\infty]$ (`proximal`): Proper, closed, $\mu$-strongly convex ($\mu \ge 0$) with cheap $\operatorname{prox}_{\gamma h}$ and domain projection $\Pi_{\operatorname{dom}(h)}$.
- **Certified Output:** Refined pair $(\hat{x}, \hat{v})$ satisfying $\hat{v} \in \nabla f(\hat{x}) + \partial h(\hat{x})$ and $\|\hat{v}\| \le \rho$.

### 2.2. Set / Cone-Constrained Composite Problem
$$
\min_{x \in \mathcal{X}} \; f(x) + h(x) \quad \text{s.t.} \quad \mathcal{A}(x) \in \mathcal{C}
$$
- $\mathcal{A}: \mathcal{X} \to \mathcal{Y}$: Linear map between PyTrees (dense/sparse matrix, `(matvec, rmatvec)` pair, or forward-only callable bound lazily to `x0` via `jax.linear_transpose`).
- $\mathcal{C} \subseteq \mathcal{Y}$: **Any closed convex set** with Euclidean projection $\Pi_{\mathcal{C}}$ (singletons $\{b\}$ for $\mathcal{A}(x) = b$, halfspaces $(-\infty, b]$, boxes $[l, u]$, SOC, PSD cone, or named PyTree dicts via `ProductConvexSet`).
- **Unified Penalty / AL Smooth Term** (for multiplier $\bar{p}$ and penalty $c > 0$, with $\bar{p} = 0$ for pure quadratic penalty in `QPAIPP` and $\bar{p} = (1-\theta)p$ for ALM variants):
  $$
  \psi_c(x; \bar{p}) = \frac{c}{2}\left\|w(x) - \Pi_{\mathcal{C}}(w(x))\right\|^2 - \frac{\|\bar{p}\|^2}{2c}, \qquad w(x) := \mathcal{A}(x) + \frac{\bar{p}}{c}
  $$
  with smooth gradient $\nabla_x \psi_c(x; \bar{p}) = \mathcal{A}^* \hat{p}(x)$, where $\hat{p}(x) := c\big(w(x) - \Pi_{\mathcal{C}}(w(x))\big) \in \mathcal{N}_{\mathcal{C}}\big(\Pi_{\mathcal{C}}(w(x))\big)$.
- **Certified KKT Output:** $(\hat{x}, \hat{p}, \hat{v})$ with $\hat{v} \in \nabla f(\hat{x}) + \partial h(\hat{x}) + \mathcal{A}^* \hat{p}$, $\|\hat{v}\| \le \rho$, and $\operatorname{dist}_{\mathcal{C}}(\mathcal{A}(\hat{x})) = \|\mathcal{A}(\hat{x}) - \Pi_{\mathcal{C}}(\mathcal{A}(\hat{x}))\| \le \eta$.

---

## 3. Key Architectural Safeguards

1. **Zero Re-Compilation Oracles:** `ProxRegularizedOracle` and `PenalizedOracle` are `@jax.tree_util.register_dataclass` structs with static callables in `meta_fields` and dynamic JAX arrays (`center`, `smooth_reg`, `prox_reg`, `multiplier`, `penalty`) in `data_fields`.
2. **Safe Curvature Splitting:** `ProxRegularizedOracle` splits $\frac{1}{\lambda} = \alpha_{\text{smooth}} + \mu_{\text{prox}}$ explicitly—never claiming $\mu_{\text{prox}} > 0$ unless `curvature_lower` $m$ is known or validated by adaptive descent/convexity checks.
3. **`vmap`-Fast Nested Loops vs. Optax Adapter:** `proxax.solve(..., jit=True)` uses nested `jax.lax.while_loop`s so `refine` only runs once per outer step even under `jax.vmap`. `proxax.optax_compat` provides the flattened per-inner-step `GradientTransformationExtraArgs` state machine while keeping `params` equal to the feasible primal iterate $y_t$.
4. **Safe Initialization & Flexible Stopping:** `solver.init` projects `x0` onto $\operatorname{dom}(h)$ via `proximal.domain_project(x0)` before any gradient evaluation, and `StoppingConfig` supports both $\ell_2$ and $\ell_\infty$ norms plus optional relative tolerances.

---

## 4. MVP Execution Milestones (Atomic Git Commits)

1. **Milestone 1 — Tooling & Core Math Primitives:** `_typing.py`, `_loop.py`, `tree.py`, `prox/`, `constraints/`, `oracle/`, `certificates/refine.py`, `problem.py` + unit tests.
2. **Milestone 2 — `ACG` & Unconstrained `AIPP`:** `solvers/base.py`, `criteria.py`, `inner/acg.py`, `unconstrained/aipp.py`, `proxax.solve()` + nonconvex composite tests (`jit=True`, `jit=False`, `vmap`).
3. **Milestone 3 — Constrained `QPAIPP` & Optax State Machine:** `constrained/qp_aipp.py`, `optax_compat/transforms.py` + constrained nonconvex QP and PyTree Optax training-loop tests.
