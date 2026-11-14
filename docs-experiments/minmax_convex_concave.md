# Convex–Concave Min–Max Benchmarks (`proxax` vs. `Clarabel`, `SCS`, `CVXPY`)

CPU (`float64`) comparison of `proxax` primal–dual min–max solvers (`PDHG`, `ExtraGradient`, `ProxPointMinMax`) against dualized conic reformulations solved by `Clarabel` (IPM), `SCS` (direct `QDLDL` and indirect `CG` ADMM), and `CVXPY (Clarabel)` (`tol = 1e-4`).

## Problem Classes

1. **Zero-Sum Matrix Game (`Simplex` $\times$ `Simplex`)** —
   $$\min_{x \in \Delta_n} \max_{y \in \Delta_m} \; x^\top A y \qquad \Longleftrightarrow \qquad \min_{x \in \Delta_n,\; t \in \mathbb{R}} \; t \quad \text{s.t.} \quad A^\top x \le t \mathbf{1}_m$$
   Tested at $(n, m) \in \{(32, 32),\; (128, 128),\; (256, 256)\}$.

2. **Regularized Bilinear Saddle-Point QP (`Box` $\times$ `Simplex`)** —
   $$\min_{x \in [-1, 1]^n} \max_{y \in \Delta_m} \; \frac{1}{2} x^\top Q_x x + x^\top K y - \frac{1}{2} y^\top Q_y y + c_x^\top x - c_y^\top y \qquad (Q_x \succeq 0,\; Q_y \succ 0)$$
   Dualized to a QP over $(x, z, \nu, \mu)$ with $Q_y = L_y L_y^\top$ for `Clarabel`, `SCS`, and `CVXPY`.

## Setup

* **Precision & Device:** `float64` (`JAX_ENABLE_X64=1`), CPU.
* **Tolerances:** `opt_tol = 1e-4` for `proxax` (`PDHG`, `ExtraGradient`, `ProxPointMinMax`); `tol_gap_abs = tol_gap_rel = tol_feas = 1e-4` for `Clarabel`; `eps_abs = eps_rel = 1e-4` for `SCS`.
* **Reproduction:**

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m benchmarks.external.minmax_convex_concave --include-cvxpy
```

## Results

### Runtime Scaling

![Convex-Concave MinMax vs. Dualized Conic Solvers](_static/plots/minmax_convex_concave.svg)

### Summary Table (`tol = 1e-4`)

| Problem | Solver | Status | Objective | Opt Res | Grad Evals | Run (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| `Matrix Game (32x32)` | `PDHG` | `CONVERGED` | 0.0000 | 9.69e-05 | 3,970 | 6.89 |
| `Matrix Game (32x32)` | `ExtraGradient` | `CONVERGED` | 0.0000 | 9.95e-05 | 4,039 | 37.62 |
| `Matrix Game (32x32)` | `ProxPointMinMax` | `CONVERGED` | 0.0000 | 7.27e-05 | 5,512 | 17.38 |
| `Matrix Game (32x32)` | `Clarabel` | `CONVERGED` | 0.0000 | 2.78e-05 | — | 0.69 |
| `Matrix Game (32x32)` | `SCS (direct)` | `CONVERGED` | 0.0000 | 2.05e-05 | — | **0.54** |
| `Matrix Game (32x32)` | `SCS (indirect)` | `CONVERGED` | 0.0000 | 1.69e-05 | — | 3.21 |
| `Matrix Game (32x32)` | `CVXPY (Clarabel)` | `CONVERGED` | 0.0000 | — | — | 4.78 |
| `Matrix Game (128x128)` | `PDHG` | `CONVERGED` | -0.0004 | 8.87e-05 | 1,442 | 10.50 |
| `Matrix Game (128x128)` | `ExtraGradient` | `CONVERGED` | -0.0004 | 9.98e-05 | 987 | 29.14 |
| `Matrix Game (128x128)` | `ProxPointMinMax` | `CONVERGED` | -0.0004 | 9.87e-05 | 11,069 | 243.82 |
| `Matrix Game (128x128)` | `Clarabel` | `CONVERGED` | -0.0004 | 2.72e-05 | — | 18.56 |
| `Matrix Game (128x128)` | `SCS (direct)` | `CONVERGED` | -0.0004 | 1.17e-04 | — | **8.25** |
| `Matrix Game (128x128)` | `SCS (indirect)` | `CONVERGED` | -0.0004 | 1.50e-04 | — | 78.29 |
| `Matrix Game (128x128)` | `CVXPY (Clarabel)` | `CONVERGED` | -0.0004 | — | — | 19.23 |
| `Matrix Game (256x256)` | `PDHG` | `CONVERGED` | -0.0001 | 9.88e-05 | 1,250 | **20.73** |
| `Matrix Game (256x256)` | `ExtraGradient` | `CONVERGED` | -0.0001 | 9.99e-05 | 1,006 | 56.44 |
| `Matrix Game (256x256)` | `ProxPointMinMax` | `CONVERGED` | -0.0001 | 8.96e-05 | 8,501 | 393.04 |
| `Matrix Game (256x256)` | `Clarabel` | `CONVERGED` | -0.0001 | 9.36e-05 | — | 176.37 |
| `Matrix Game (256x256)` | `SCS (direct)` | `CONVERGED` | -0.0001 | 9.42e-05 | — | 53.13 |
| `Matrix Game (256x256)` | `SCS (indirect)` | `CONVERGED` | -0.0001 | 1.29e-04 | — | 695.78 |
| `Matrix Game (256x256)` | `CVXPY (Clarabel)` | `CONVERGED` | -0.0001 | — | — | 258.03 |
| `Bilinear Saddle QP (32x32)` | `PDHG` | `CONVERGED` | -2.4130 | 7.94e-05 | 386 | 0.97 |
| `Bilinear Saddle QP (32x32)` | `ExtraGradient` | `CONVERGED` | -2.4130 | 9.81e-05 | 454 | 3.50 |
| `Bilinear Saddle QP (32x32)` | `ProxPointMinMax` | `CONVERGED` | -2.4130 | 9.25e-05 | 3,928 | 8.96 |
| `Bilinear Saddle QP (32x32)` | `Clarabel` | `CONVERGED` | -2.4130 | 8.99e-10 | — | 0.97 |
| `Bilinear Saddle QP (32x32)` | `SCS (direct)` | `CONVERGED` | -2.4130 | 2.66e-07 | — | **0.64** |
| `Bilinear Saddle QP (32x32)` | `SCS (indirect)` | `CONVERGED` | -2.4130 | 4.51e-06 | — | 8.44 |
| `Bilinear Saddle QP (32x32)` | `CVXPY (Clarabel)` | `CONVERGED` | -2.4130 | — | — | 6.00 |
| `Bilinear Saddle QP (128x128)` | `PDHG` | `CONVERGED` | -11.8179 | 5.52e-05 | 450 | 6.18 |
| `Bilinear Saddle QP (128x128)` | `ExtraGradient` | `CONVERGED` | -11.8179 | 9.78e-05 | 492 | 14.91 |
| `Bilinear Saddle QP (128x128)` | `ProxPointMinMax` | `CONVERGED` | -11.8179 | 9.06e-05 | 4,471 | 91.01 |
| `Bilinear Saddle QP (128x128)` | `Clarabel` | `CONVERGED` | -11.8178 | 1.78e-08 | — | 38.75 |
| `Bilinear Saddle QP (128x128)` | `SCS (direct)` | `CONVERGED` | -11.8179 | 3.04e-06 | — | **5.23** |
| `Bilinear Saddle QP (128x128)` | `SCS (indirect)` | `CONVERGED` | -11.8179 | 7.27e-06 | — | 114.03 |
| `Bilinear Saddle QP (128x128)` | `CVXPY (Clarabel)` | `CONVERGED` | -11.8179 | — | — | 66.32 |
| `Bilinear Saddle QP (256x256)` | `PDHG` | `CONVERGED` | -21.5770 | 7.66e-05 | 418 | **12.80** |
| `Bilinear Saddle QP (256x256)` | `ExtraGradient` | `CONVERGED` | -21.5770 | 9.96e-05 | 541 | 35.78 |
| `Bilinear Saddle QP (256x256)` | `ProxPointMinMax` | `CONVERGED` | -21.5770 | 9.59e-05 | 4,995 | 286.35 |
| `Bilinear Saddle QP (256x256)` | `Clarabel` | `CONVERGED` | -21.5770 | 1.85e-08 | — | 265.58 |
| `Bilinear Saddle QP (256x256)` | `SCS (direct)` | `CONVERGED` | -21.5770 | 2.46e-06 | — | 28.16 |
| `Bilinear Saddle QP (256x256)` | `SCS (indirect)` | `CONVERGED` | -21.5770 | 1.06e-05 | — | 655.56 |
| `Bilinear Saddle QP (256x256)` | `CVXPY (Clarabel)` | `CONVERGED` | -21.5770 | — | — | 408.41 |

## Key Observations

1. **All solvers converge to identical saddle-point values** across all 6 problems (e.g., `-2.4130`, `-11.8179`, `-21.5770` on `Bilinear Saddle QP`).
2. **`PDHG` leads at scale (`256x256`):** By avoiding dualization and dense KKT factorizations, `PDHG` solves `Matrix Game (256x256)` in **20.73 ms** (**2.6×** faster than `SCS (direct)` at `53.13 ms`, **8.5×** faster than `Clarabel` at `176.37 ms`, **12.4×** faster than `CVXPY`) and `Bilinear Saddle QP (256x256)` in **12.80 ms** (**2.2×** faster than `SCS (direct)` at `28.16 ms`, **20.7×** faster than `Clarabel` at `265.58 ms`).
3. **Dense dualization hurts IPMs and indirect CG:** Dualizing `Bilinear Saddle QP` introduces a dense $m \times n$ block plus Cholesky factor $L_y$, making `Clarabel` scale cubically (`0.97 ms` $\to$ `38.75 ms` $\to$ `265.58 ms`) and `SCS (indirect)` slow down (`655.56 ms` at `256x256`), whereas `PDHG` only needs cheap matrix–vector products.
