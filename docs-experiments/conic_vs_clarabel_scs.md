# Conic Optimization Benchmarks (`proxax` vs. `Clarabel`, `SCS`, `CVXPY`)

CPU (`float64`) comparison of `proxax` constrained solvers (`IALM (APD)`, `PANOC-ALM`, `QP-APD`, `QP-AIPP`) against `Clarabel` (IPM), `SCS` (direct `QDLDL` and indirect `CG` ADMM), and `CVXPY (SCS)` across Second-Order Cone (`SOC`), Exponential Cone (`ExpCone`), and Spectrahedron SDP (`Spectrahedron`) problems (`tol = 1e-4`).

## Problem Classes

1. **SOC Portfolio (`SOCP`, $n \in \{32,\; 128,\; 512\}$)** —
   $$\min_{x \in \mathcal{K}_{\text{SOC}}^n} \; \frac{1}{2} x^\top \Sigma x - \mu^\top x \quad \text{s.t.} \quad x_0 = 2,\; \mathbf{1}^\top x = 3$$
2. **Exponential Cone Blocks (`ExpCone`, $k \in \{1,\; 16,\; 64\}$ blocks, $n = 3k$)** —
   $$\min_{x^{(j)} \in \mathcal{K}_{\exp}} \; \sum_{j=1}^k \left(\frac{1}{2}\|x^{(j)} - v^{(j)}\|_2^2 + (c^{(j)})^\top x^{(j)}\right) \quad \text{s.t.} \quad x_0^{(j)} + x_1^{(j)} = 2$$
   (`proxax` vectorizes the $k$ projections via `jax.vmap` over a $(k, 3)$ array).
3. **Spectrahedron SDP (`SDP`, $n \in \{8,\; 24,\; 48\}$)** —
   $$\min_{X \succeq 0,\; \operatorname{Tr}(X)=1} \; \frac{1}{2}\|X - M\|_F^2 \quad \text{s.t.} \quad X_{00} + X_{11} = 0.5$$

## Setup

* **Precision & Device:** `float64` (`JAX_ENABLE_X64=1`), CPU.
* **Tolerances & Limits:** `opt_tol = feas_tol = 1e-4` (`time_limit = 5.0 s`).
* **Reproduction:**

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m benchmarks.external.conic_vs_clarabel_scs --time-limit 5.0 --include-cvxpy
```

## Results

### Runtime Scaling

![Conic Optimization Runtime Scaling](_static/plots/conic_vs_clarabel_scs.svg)

### Dolan–Moré Runtime Profile

![Conic Optimization Dolan-Moré Profile](_static/plots/conic_vs_clarabel_scs_profile.svg)

### Summary Table (`tol = 1e-4`)

| Problem | Solver | Status | Objective | Opt Res | Feas Res | Grad Evals | Run (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `SOC Portfolio (n=32)` | `PANOC-ALM` | `CONVERGED` | -0.3392 | 7.40e-05 | 2.89e-05 | 615 | 3.27 |
| `SOC Portfolio (n=32)` | `IALM (APD)` | `CONVERGED` | -0.3392 | 9.14e-05 | 5.27e-05 | 1,474 | 1.45 |
| `SOC Portfolio (n=32)` | `QP-APD` | `CONVERGED` | -0.3392 | 1.00e-04 | 5.90e-05 | 11,018 | 4.73 |
| `SOC Portfolio (n=32)` | `QP-AIPP` | `CONVERGED` | -0.3392 | 6.47e-05 | 5.90e-05 | 19,352 | 13.77 |
| `SOC Portfolio (n=32)` | `Clarabel` | `CONVERGED` | -0.3392 | 1.10e-07 | 9.70e-16 | — | 0.41 |
| `SOC Portfolio (n=32)` | `SCS (direct)` | `CONVERGED` | -0.3392 | 6.02e-08 | 6.63e-08 | — | **0.30** |
| `SOC Portfolio (n=32)` | `SCS (indirect)` | `CONVERGED` | -0.3392 | 4.30e-06 | 2.93e-07 | — | 0.97 |
| `SOC Portfolio (n=32)` | `CVXPY (SCS)` | `CONVERGED` | -0.3392 | — | — | — | 6.86 |
| `SOC Portfolio (n=128)` | `PANOC-ALM` | `CONVERGED` | -0.4193 | 9.32e-05 | 6.37e-05 | 1,126 | 9.24 |
| `SOC Portfolio (n=128)` | `IALM (APD)` | `CONVERGED` | -0.4193 | 9.17e-05 | 8.10e-05 | 2,982 | 7.49 |
| `SOC Portfolio (n=128)` | `QP-APD` | `CONVERGED` | -0.4193 | 9.93e-05 | 8.15e-05 | 18,734 | 39.61 |
| `SOC Portfolio (n=128)` | `QP-AIPP` | `CONVERGED` | -0.4193 | 9.35e-05 | 8.15e-05 | 57,184 | 205.11 |
| `SOC Portfolio (n=128)` | `Clarabel` | `CONVERGED` | -0.4193 | 3.77e-11 | 3.41e-16 | — | 9.95 |
| `SOC Portfolio (n=128)` | `SCS (direct)` | `CONVERGED` | -0.4193 | 9.80e-08 | 8.47e-08 | — | **1.32** |
| `SOC Portfolio (n=128)` | `SCS (indirect)` | `CONVERGED` | -0.4193 | 1.04e-05 | 4.72e-07 | — | 6.15 |
| `SOC Portfolio (n=128)` | `CVXPY (SCS)` | `CONVERGED` | -0.4193 | — | — | — | 7.90 |
| `SOC Portfolio (n=512)` | `PANOC-ALM` | `CONVERGED` | -1.6135 | 8.42e-05 | 2.72e-05 | 2,394 | 112.66 |
| `SOC Portfolio (n=512)` | `IALM (APD)` | `CONVERGED` | -1.6135 | 9.84e-05 | 4.95e-05 | 5,458 | 173.87 |
| `SOC Portfolio (n=512)` | `QP-APD` | `CONVERGED` | -1.6135 | 1.00e-04 | 5.78e-05 | 134,138 | 4,171.45 |
| `SOC Portfolio (n=512)` | `QP-AIPP` | `TIME_LIMIT_REACHED` | -1.6136 | 1.68e-04 | 4.62e-04 | 87,021 | 5,014.74 |
| `SOC Portfolio (n=512)` | `Clarabel` | `CONVERGED` | -1.6135 | 7.15e-09 | 1.45e-14 | — | 86.38 |
| `SOC Portfolio (n=512)` | `SCS (direct)` | `CONVERGED` | -1.6135 | 2.58e-08 | 9.52e-08 | — | **23.17** |
| `SOC Portfolio (n=512)` | `SCS (indirect)` | `CONVERGED` | -1.6135 | 4.77e-07 | 9.50e-08 | — | 141.32 |
| `SOC Portfolio (n=512)` | `CVXPY (SCS)` | `CONVERGED` | -1.6135 | — | — | — | 72.55 |
| `ExpCone Blocks (k=1, n=3)` | `PANOC-ALM` | `CONVERGED` | 1.1403 | 9.92e-05 | 8.66e-05 | 169 | 1.29 |
| `ExpCone Blocks (k=1, n=3)` | `IALM (APD)` | `CONVERGED` | 1.1403 | 3.86e-05 | 4.97e-05 | 158 | 1.23 |
| `ExpCone Blocks (k=1, n=3)` | `QP-APD` | `CONVERGED` | 1.1403 | 9.97e-05 | 5.90e-05 | 492 | 1.44 |
| `ExpCone Blocks (k=1, n=3)` | `QP-AIPP` | `CONVERGED` | 1.1403 | 1.04e-05 | 5.90e-05 | 1,797 | 3.97 |
| `ExpCone Blocks (k=1, n=3)` | `Clarabel` | `CONVERGED` | 1.1403 | 3.84e-12 | 1.95e-12 | — | 0.16 |
| `ExpCone Blocks (k=1, n=3)` | `SCS (direct)` | `CONVERGED` | 1.1403 | 6.31e-08 | 1.76e-07 | — | 0.23 |
| `ExpCone Blocks (k=1, n=3)` | `SCS (indirect)` | `CONVERGED` | 1.1403 | 6.31e-08 | 1.76e-07 | — | **0.14** |
| `ExpCone Blocks (k=1, n=3)` | `CVXPY (SCS)` | `CONVERGED` | 1.1403 | — | — | — | 4.10 |
| `ExpCone Blocks (k=16, n=48)` | `PANOC-ALM` | `CONVERGED` | 20.0304 | 1.89e-05 | 5.36e-05 | 339 | 5.46 |
| `ExpCone Blocks (k=16, n=48)` | `IALM (APD)` | `CONVERGED` | 20.0303 | 9.42e-05 | 5.92e-05 | 276 | 2.49 |
| `ExpCone Blocks (k=16, n=48)` | `QP-APD` | `CONVERGED` | 20.0303 | 9.99e-05 | 7.59e-05 | 3,144 | 18.20 |
| `ExpCone Blocks (k=16, n=48)` | `QP-AIPP` | `CONVERGED` | 20.0303 | 1.94e-05 | 7.59e-05 | 6,725 | 73.32 |
| `ExpCone Blocks (k=16, n=48)` | `Clarabel` | `CONVERGED` | 20.0311 | 2.00e-12 | 9.05e-13 | — | 0.63 |
| `ExpCone Blocks (k=16, n=48)` | `SCS (direct)` | `CONVERGED` | 20.0306 | 4.16e-07 | 3.69e-05 | — | 0.68 |
| `ExpCone Blocks (k=16, n=48)` | `SCS (indirect)` | `CONVERGED` | 20.0306 | 5.69e-07 | 3.83e-05 | — | **0.62** |
| `ExpCone Blocks (k=16, n=48)` | `CVXPY (SCS)` | `CONVERGED` | 20.0306 | — | — | — | 4.65 |
| `ExpCone Blocks (k=64, n=192)` | `PANOC-ALM` | `CONVERGED` | 78.2354 | 1.10e-05 | 3.09e-05 | 361 | 7.73 |
| `ExpCone Blocks (k=64, n=192)` | `IALM (APD)` | `CONVERGED` | 78.2353 | 5.42e-05 | 3.42e-05 | 310 | 4.74 |
| `ExpCone Blocks (k=64, n=192)` | `QP-APD` | `CONVERGED` | 78.2350 | 9.99e-05 | 7.20e-05 | 5,144 | 39.12 |
| `ExpCone Blocks (k=64, n=192)` | `QP-AIPP` | `CONVERGED` | 78.2350 | 5.68e-06 | 7.20e-05 | 9,004 | 130.10 |
| `ExpCone Blocks (k=64, n=192)` | `Clarabel` | `CONVERGED` | 78.2381 | 2.64e-12 | 1.26e-12 | — | **0.91** |
| `ExpCone Blocks (k=64, n=192)` | `SCS (direct)` | `CONVERGED` | 78.2357 | 3.92e-07 | 3.66e-05 | — | 1.22 |
| `ExpCone Blocks (k=64, n=192)` | `SCS (indirect)` | `CONVERGED` | 78.2357 | 1.48e-06 | 3.82e-05 | — | 1.58 |
| `ExpCone Blocks (k=64, n=192)` | `CVXPY (SCS)` | `CONVERGED` | 78.2357 | — | — | — | 6.17 |
| `Spectrahedron SDP (n=8)` | `PANOC-ALM` | `CONVERGED` | 13.4554 | 3.97e-05 | 8.34e-05 | 301 | 4.63 |
| `Spectrahedron SDP (n=8)` | `IALM (APD)` | `CONVERGED` | 13.4554 | 4.26e-05 | 8.69e-05 | 222 | 1.74 |
| `Spectrahedron SDP (n=8)` | `QP-APD` | `CONVERGED` | 13.4554 | 9.79e-05 | 9.76e-05 | 2,650 | 8.32 |
| `Spectrahedron SDP (n=8)` | `QP-AIPP` | `CONVERGED` | 13.4554 | 2.56e-05 | 9.76e-05 | 4,488 | 26.01 |
| `Spectrahedron SDP (n=8)` | `Clarabel` | `CONVERGED` | 13.4556 | 9.13e-09 | 1.82e-08 | — | **0.47** |
| `Spectrahedron SDP (n=8)` | `SCS (direct)` | `CONVERGED` | 13.4556 | 1.65e-07 | 2.50e-05 | — | 0.70 |
| `Spectrahedron SDP (n=8)` | `SCS (indirect)` | `CONVERGED` | 13.4556 | 5.42e-06 | 2.51e-05 | — | 0.75 |
| `Spectrahedron SDP (n=8)` | `CVXPY (SCS)` | `CONVERGED` | 13.4556 | — | — | — | 6.55 |
| `Spectrahedron SDP (n=24)` | `PANOC-ALM` | `CONVERGED` | 140.0054 | 9.89e-05 | 5.85e-05 | 527 | 20.79 |
| `Spectrahedron SDP (n=24)` | `IALM (APD)` | `CONVERGED` | 140.0054 | 6.31e-05 | 5.62e-05 | 298 | 9.29 |
| `Spectrahedron SDP (n=24)` | `QP-APD` | `CONVERGED` | 140.0054 | 9.98e-05 | 7.30e-05 | 5,906 | 119.75 |
| `Spectrahedron SDP (n=24)` | `QP-AIPP` | `CONVERGED` | 140.0054 | 2.34e-05 | 7.30e-05 | 7,848 | 308.30 |
| `Spectrahedron SDP (n=24)` | `Clarabel` | `CONVERGED` | 140.0060 | 1.06e-09 | 4.13e-09 | — | 24.32 |
| `Spectrahedron SDP (n=24)` | `SCS (direct)` | `CONVERGED` | 140.0056 | 5.90e-06 | 2.64e-05 | — | **5.25** |
| `Spectrahedron SDP (n=24)` | `SCS (indirect)` | `CONVERGED` | 140.0056 | 5.92e-06 | 2.61e-05 | — | 6.06 |
| `Spectrahedron SDP (n=24)` | `CVXPY (SCS)` | `CONVERGED` | 140.0058 | — | — | — | 11.91 |
| `Spectrahedron SDP (n=48)` | `PANOC-ALM` | `CONVERGED` | 599.5343 | 8.54e-05 | 7.90e-05 | 621 | 258.68 |
| `Spectrahedron SDP (n=48)` | `IALM (APD)` | `CONVERGED` | 599.5343 | 9.49e-05 | 7.70e-05 | 384 | 71.80 |
| `Spectrahedron SDP (n=48)` | `QP-APD` | `CONVERGED` | 599.5345 | 9.98e-05 | 5.37e-05 | 6,506 | 1,801.25 |
| `Spectrahedron SDP (n=48)` | `QP-AIPP` | `TIME_LIMIT_REACHED` | 599.4411 | 2.03e-03 | 1.35e-02 | 1,138 | 5,366.10 |
| `Spectrahedron SDP (n=48)` | `Clarabel` | `CONVERGED` | 599.5352 | 2.47e-07 | 1.71e-06 | — | 846.55 |
| `Spectrahedron SDP (n=48)` | `SCS (direct)` | `CONVERGED` | 599.5349 | 5.48e-11 | 3.13e-11 | — | **29.72** |
| `Spectrahedron SDP (n=48)` | `SCS (indirect)` | `CONVERGED` | 599.5349 | 7.32e-07 | 2.91e-07 | — | 36.36 |
| `Spectrahedron SDP (n=48)` | `CVXPY (SCS)` | `CONVERGED` | 599.5346 | — | — | — | 35.22 |

## Key Observations

1. **`IALM (APD)` beats `Clarabel` by 2.6×–11.8× on `Spectrahedron SDP` (`n >= 24`):** At `n=48`, `Clarabel`'s interior-point KKT factorization over `48*49/2 = 1,176` `svec` variables takes **846.55 ms**, whereas `IALM (APD)` solves the matrix-space problem in **71.80 ms** (**11.8× faster**). `SCS (direct)` is fastest overall on single CPU SDP (**29.72 ms** at `n=48`) via C/LAPACK `syevd` + Anderson acceleration.
2. **`IALM (APD)` and `PANOC-ALM` vastly outperform `QP-APD` / `QP-AIPP` on conic problems:** On `SOC Portfolio (n=512)` and `Spectrahedron SDP (n=48)`, `IALM`/`PANOC-ALM` converge in **71.8–258.7 ms**, while `QP-APD` takes **1.8–4.2 s** and `QP-AIPP` hits the `5 s` limit.
3. **`vmap` keeps `ExpCone` scaling nearly flat:** Vectorizing `ExpCone` projections over `(k, 3)` lets `IALM (APD)` scale from **1.23 ms** (`k=1`) to **4.74 ms** (`k=64`), beating `CVXPY (SCS)` (**6.17 ms**) at `k=64`, while `Clarabel` (`0.91 ms`) and `SCS` (`1.22 ms`) remain fastest on small unbatched `ExpCone` blocks.
