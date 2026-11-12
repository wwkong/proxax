# `proxax` — Workspace Rules & Agent Context

`proxax` (https://github.com/wwkong/proxax) is a Python library of **certified first-order proximal point methods** for unconstrained and set/cone-constrained nonconvex composite optimization in **JAX** and **Optax**, inspired by William Kong's MATLAB framework **[`wwkong/nc_opt`](https://github.com/wwkong/nc_opt)**.

Read `DESIGN.md` for mathematical formulations and `docs/nc_opt_reference.md` for the detailed mapping to `wwkong/nc_opt` algorithms, papers, and Phase 2+ roadmap.

---

## 1. Environment & Verification Commands (WSL2 Primary Workspace)

- **Canonical Repo Path:** `/home/wkong/Projects/proxax` (`origin`: `https://github.com/wwkong/proxax.git`, branch `main`).
- **Package Manager & Runtime:** `uv` (`pyproject.toml` + `uv.lock`) with Python 3.14, `jax[cuda13]`, and `optax`.
- **Runtime Dependency Rule:** `src/proxax/` must depend **strictly on `jax` and `optax` only** (zero SciPy / CVXPY / Equinox / Flax runtime dependencies).
- **GPU Memory Rule (12 GB RTX 3060 shared with Windows host):**
  - Always ensure `XLA_PYTHON_CLIENT_PREALLOCATE=false` when running JAX on GPU (`tests/conftest.py` and `docs/conf.py` set this automatically).
  - **Keep unit tests tiny, fast (< 10s total), and low-memory** (2D–4D problems, tight iteration caps, `jax.lax.while_loop` over long Python dispatch loops). Never introduce large matrices or long-running benchmarks into `tests/`.

Run all four verification gates via `uv` before committing:

```bash
uv run ruff check .
uv run mypy
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run pytest -v
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run sphinx-build -W -b html docs site
```

---

## 2. Strict Coding & Style Conventions (Google Python Style Guide)

1. **80-Column Line Limit:** Enforced by `ruff` (`line-length = 80`, `indent-width = 2` or standard 2/4-space blocks; match existing module style and pass `uv run ruff check .`).
2. **Module-Only Imports (`TID253` / Google Style):**
   - Always import modules, never classes/functions directly (except `typing` / `collections.abc` when module-qualified):
     - Good: `import dataclasses`, `import typing`, `from collections import abc`, `from proxax import tree`, `from proxax.prox import base as prox_base`.
     - Bad: `from dataclasses import dataclass`, `from proxax.prox.base import ProxOperator`.
   - Avoid `B008` function calls in default parameter values (bind module-level finals like `_ZERO_PROX: typing.Final = prox_base.Zero()`).
3. **Immutable PyTree Dataclasses & Strict Typing (`mypy --strict`):**
   - Use `@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)` for solvers, configs, problems, states, and results.
   - When registering dataclasses with `meta_fields` / `data_fields`, call `jax.tree_util.register_dataclass(Cls, data_fields=[...], meta_fields=[...])` after the class definition so `mypy --strict` resolves types cleanly against JAX stubs.
   - Use `tree.eps(dtype)` (`float(np.finfo(dtype).eps)`) instead of untyped `jnp.finfo(dtype).eps`.
4. **Sphinx / Google Docstrings:**
   - Avoid trailing `'s` attached directly to single-backtick inline literals (write `` `Constraint` objects `` instead of `` `Constraint`s ``) and keep blank lines around bullet/formula blocks so `sphinx-build -W` passes with zero warnings.

---

## 3. Core Mathematical & Numerical Invariants (Do Not Break)

1. **Exact Stationarity Certificates (`src/proxax/certificates/stationarity.py`):**
   - Every solver returns a certified refined point `x_hat = prox_{(1/L) h}(y - (1/L) grad_f(y))` and exact composite subgradient `v_hat = grad_f(x_hat) - grad_f(y) + L * (y - x_hat) in grad_f(x_hat) + dh(x_hat)` (plus `A^*(p_hat)` in constrained problems), never raw unrefined inner iterates.
2. **Strongly-Convex 1-Prox `ACG` (`src/proxax/solvers/inner/acg.py`):**
   - Maintains the exact $\mu$-strongly convex estimating sequence with $c_\mu = 1 + \mu A_t$, $b_\mu = 1 + 2\mu A_t$, $d_\mu = 1 + \mu A_{t+1}$, convex combination weights $(A_t d_\mu, a_t c_\mu)$, and exact $\varepsilon$-subgradient pair $(u_{t+1}, \eta_{t+1}) \in \partial_{\eta_{t+1}} \psi(y_{t+1})$.
3. **IEEE `float32` Adaptive Smoothness Noise-Floor Guard:**
   - Only relax `L_next = L_used * ls_decrease` when `0.5 * L_used * ||y_next - x_tilde||^2 > 100 * eps * (1 + |f|)`. Without this guard, `L` decays toward `0` when step norms fall below $\sqrt{\epsilon_{\text{mach}}} \approx 3.5 \times 10^{-4}$ in `float32`.
   - When `lower_curvature == 0.0` (convex `f`), calibrate default `lam0 = 10.0 / max(L0, 1e-6)` rather than huge values like `1e3` so `1 / lam` stays well above the `float32` precision floor in the $\sigma$-criterion.
4. **Curvature Preservation Across Penalty / ALM Stages (`src/proxax/oracle/composite.py`):**
   - `PenalizedOracle` adds the convex Moreau envelope $\psi_c(x; \bar{p}) = \frac{c}{2}\|w(x) - \Pi_{\mathcal{C}}(w(x))\|_2^2 - \frac{\|\bar{p}\|_2^2}{2c}$ where $w(x) = \mathcal{A}(x) + \bar{p}/c$ and $\bar{p} = (1 - \theta) p$. Because $\psi_c$ is convex, `curvature_lower` $m$ of $f$ is unchanged as $c_k$ grows, so proximal steps $\lambda_k$ must **never** shrink as $O(1/c_k)$.
