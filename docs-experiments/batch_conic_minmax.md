# Batched Conic & Min–Max Solves (`proxax` vs. `Clarabel`, `SCS`)

CPU (`float64`) comparison of `px.batch_solve(..., backend="threads")` against `ThreadPoolExecutor` loops over `Clarabel` and `SCS (direct)` for batch sizes $B \in \{1,\; 16,\; 64\}$ (`tol = 1e-4`).

## Problem Classes

1. **Batched SOC Portfolio (`n = 64`, $B \in \{1, 16, 64\}$)** —
   Varying linear return vector $\mu^{(b)}$ over `px.prox.SOC()` with two linear equality constraints (`proxax-IALM (threads)` vs. `Clarabel` & `SCS`).
2. **Batched Regularized Bilinear Saddle QP (`n = 64`, $B \in \{1, 16, 64\}$)** —
   Varying linear terms $(c_x^{(b)}, c_y^{(b)})$ over `Box(-1, 1)` $\times$ `Simplex` (`proxax-PDHG (threads)` vs. dualized `Clarabel` & `SCS`).

## Setup

* **Precision & Device:** `float64` (`JAX_ENABLE_X64=1`), CPU (`threads` backend).
* **Tolerances:** `opt_tol = feas_tol = 1e-4`.
* **Reproduction:**

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m benchmarks.external.batch_conic_minmax
```

## Results

### Batch-Size Scaling

![Batched Conic and MinMax Scaling](_static/plots/batch_conic_minmax.svg)

### Summary Table (`tol = 1e-4`)

| Problem | Solver | Status | Mean Obj | Max Opt Res | Wall (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| `Batched SOCP (n=64, B=1)` | `proxax-IALM (threads)` | `CONVERGED` | -0.2087 | 9.04e-05 | 1.77 |
| `Batched SOCP (n=64, B=1)` | `Clarabel (threads)` | `CONVERGED` | — | — | 0.74 |
| `Batched SOCP (n=64, B=1)` | `SCS (threads)` | `CONVERGED` | — | — | **0.35** |
| `Batched SOCP (n=64, B=16)` | `proxax-IALM (threads)` | `CONVERGED` | -0.3926 | 9.88e-05 | 5.34 |
| `Batched SOCP (n=64, B=16)` | `Clarabel (threads)` | `CONVERGED` | — | — | 7.77 |
| `Batched SOCP (n=64, B=16)` | `SCS (threads)` | `CONVERGED` | — | — | **3.75** |
| `Batched SOCP (n=64, B=64)` | `proxax-IALM (threads)` | `CONVERGED` | -0.3576 | 1.00e-04 | 20.22 |
| `Batched SOCP (n=64, B=64)` | `Clarabel (threads)` | `CONVERGED` | — | — | 22.42 |
| `Batched SOCP (n=64, B=64)` | `SCS (threads)` | `CONVERGED` | — | — | **11.00** |
| `Batched Saddle QP (n=64, B=1)` | `proxax-PDHG (threads)` | `CONVERGED` | -4.9075 | 9.01e-05 | **1.12** |
| `Batched Saddle QP (n=64, B=1)` | `Clarabel (threads)` | `CONVERGED` | — | — | 3.70 |
| `Batched Saddle QP (n=64, B=1)` | `SCS (threads)` | `CONVERGED` | — | — | 1.38 |
| `Batched Saddle QP (n=64, B=16)` | `proxax-PDHG (threads)` | `CONVERGED` | -4.6348 | 9.30e-05 | 8.03 |
| `Batched Saddle QP (n=64, B=16)` | `Clarabel (threads)` | `CONVERGED` | — | — | 22.27 |
| `Batched Saddle QP (n=64, B=16)` | `SCS (threads)` | `CONVERGED` | — | — | **5.65** |
| `Batched Saddle QP (n=64, B=64)` | `proxax-PDHG (threads)` | `CONVERGED` | -4.8720 | 9.94e-05 | 30.85 |
| `Batched Saddle QP (n=64, B=64)` | `Clarabel (threads)` | `CONVERGED` | — | — | 65.29 |
| `Batched Saddle QP (n=64, B=64)` | `SCS (threads)` | `CONVERGED` | — | — | **16.95** |

## Key Observations

1. **Beats threaded `Clarabel` on batched SOCP and Min–Max:** `proxax-IALM (threads)` outperforms `Clarabel (threads)` at `B=16` (`5.34 ms` vs. `7.77 ms`) and `B=64` (`20.22 ms` vs. `22.42 ms`), while `proxax-PDHG (threads)` is **2.1×–3.3× faster** than `Clarabel (threads)` across all batch sizes on `Batched Saddle QP` (`30.85 ms` vs. `65.29 ms` at `B=64`).
2. **Thread-parallel scaling:** Going from `B=1` to `B=64` (64× more problems) increases `proxax-IALM` wall time by only **11.4×** (`1.77 ms` $\to$ `20.22 ms`), while `SCS (threads)` remains ~1.8× faster at `B=64` (`11.00 ms` / `16.95 ms`).
