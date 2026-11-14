# Spectral Nuclear-Norm Matrix Completion (`proxax` vs. `Clarabel`, `SCS`, `CVXPY`)

CPU (`float64`) comparison of `proxax` spectral conditional-gradient solvers (`DAICG (monotone)`, `DAICG (nonmonotone)`, `IAICG`) in $m \times n$ matrix space against lifted $(m+n) \times (m+n)$ SDP formulations in `Clarabel`, `SCS`, and `CVXPY (SCS)` (`tol = 1e-4`).

## Problem Formulation

$$\min_{X \in \mathbb{R}^{m \times n}} \; \frac{1}{2} \|P_\Omega(X - M_{\text{obs}})\|_F^2 + \lambda \|X\|_* \qquad (\text{rank}=3,\; |\Omega|/(mn) = 0.50,\; \lambda = 0.5\sqrt{\max(m, n)})$$

Tested at $(m, n) \in \{(16, 16),\; (32, 32),\; (48, 48)\}$. For `Clarabel` and `SCS`, $\|X\|_*$ is lifted to a $(m+n) \times (m+n)$ PSD cone constraint $\begin{bmatrix} W_1 & X \\ X^\top & W_2 \end{bmatrix} \succeq 0$.

## Setup

* **Precision & Device:** `float64` (`JAX_ENABLE_X64=1`), CPU.
* **Tolerances:** `opt_tol = 1e-4` (`time_limit = 15.0 s`).
* **Reproduction:**

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m benchmarks.external.spectral_vs_clarabel_scs --include-cvxpy
```

## Results

### Runtime Scaling & Dolan–Moré Profile

![Spectral Matrix Completion Scaling and Profile](_static/plots/spectral_vs_clarabel_scs.svg)

### Summary Table (`tol = 1e-4`)

| Problem | Solver | Status | Objective | Opt Res | Run (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| `Nuclear MC (16x16)` | `DAICG (monotone)` | `CONVERGED` | 97.9982 | 9.69e-05 | 16.40 |
| `Nuclear MC (16x16)` | `DAICG (nonmonotone)` | `CONVERGED` | 97.9982 | 9.45e-05 | 13.95 |
| `Nuclear MC (16x16)` | `IAICG` | `CONVERGED` | 97.9982 | 9.91e-05 | 14.83 |
| `Nuclear MC (16x16)` | `Clarabel` | `CONVERGED` | 98.0106 | 1.88e-06 | 99.59 |
| `Nuclear MC (16x16)` | `SCS (direct)` | `CONVERGED` | 97.9982 | 5.01e-07 | **3.59** |
| `Nuclear MC (16x16)` | `SCS (indirect)` | `CONVERGED` | 97.9982 | 1.49e-06 | 4.00 |
| `Nuclear MC (16x16)` | `CVXPY (SCS)` | `CONVERGED` | 98.0115 | — | 14.28 |
| `Nuclear MC (32x32)` | `DAICG (monotone)` | `CONVERGED` | 299.3581 | 9.69e-05 | 31.02 |
| `Nuclear MC (32x32)` | `DAICG (nonmonotone)` | `CONVERGED` | 299.3581 | 7.12e-05 | 30.48 |
| `Nuclear MC (32x32)` | `IAICG` | `CONVERGED` | 299.3581 | 9.44e-05 | 25.02 |
| `Nuclear MC (32x32)` | `Clarabel` | `CONVERGED` | 299.4063 | 4.62e-06 | 1,456.05 |
| `Nuclear MC (32x32)` | `SCS (direct)` | `CONVERGED` | 299.3604 | 1.17e-04 | **10.27** |
| `Nuclear MC (32x32)` | `SCS (indirect)` | `CONVERGED` | 299.3581 | 4.47e-06 | 18.01 |
| `Nuclear MC (32x32)` | `CVXPY (SCS)` | `CONVERGED` | 299.3582 | — | 28.93 |
| `Nuclear MC (48x48)` | `DAICG (monotone)` | `CONVERGED` | 553.2504 | 9.59e-05 | 302.39 |
| `Nuclear MC (48x48)` | `DAICG (nonmonotone)` | `CONVERGED` | 553.2504 | 9.21e-05 | 446.49 |
| `Nuclear MC (48x48)` | `IAICG` | `CONVERGED` | 553.2504 | 9.94e-05 | 379.58 |
| `Nuclear MC (48x48)` | `Clarabel` | `CONVERGED` | 553.2621 | 3.74e-07 | 6,242.06 |
| `Nuclear MC (48x48)` | `SCS (direct)` | `CONVERGED` | 553.2519 | 1.10e-04 | **21.02** |
| `Nuclear MC (48x48)` | `SCS (indirect)` | `CONVERGED` | 553.2504 | 1.71e-06 | 40.52 |
| `Nuclear MC (48x48)` | `CVXPY (SCS)` | `CONVERGED` | 553.2504 | — | 57.18 |

## Key Observations

1. **7.1×–58.2× faster than `Clarabel`:** Operating directly in $m \times n$ matrix space avoids lifting to $(m+n) \times (m+n)$ SDP triangles (`4,656` `svec` variables at `48x48`), making `DAICG`/`IAICG` **7.1×** faster at `16x16` (`13.95 ms` vs. `99.59 ms`), **58.2×** faster at `32x32` (`25.02 ms` vs. `1,456.05 ms`), and **20.6×** faster at `48x48` (`302.39 ms` vs. `6,242.06 ms`).
2. **`SCS (direct)` leads on convex nuclear-norm SDPs:** `SCS` leverages C/LAPACK `syevd` + Anderson acceleration to solve `48x48` in `21.02 ms`, whereas `proxax`'s `SpectralProblem` is primarily designed for *nonconvex* spectral penalties (where `Clarabel` and `SCS` cannot run).
