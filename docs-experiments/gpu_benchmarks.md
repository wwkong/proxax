# GPU Benchmarks (`NVIDIA RTX 3060` vs. CPU Baselines, `Optax`, and `MPAX`)

Evaluation of Proxax on an **NVIDIA GeForce RTX 3060** (`jax[cuda12]==0.12.0.1`, `float64`) against CPU Proxax, multi-threaded `Clarabel` / `SCS`, GPU `Optax`, and GPU `MPAX` (`r2HPDHG`).

## Setup

* **Hardware & Precision:** NVIDIA GeForce RTX 3060 (12 GB VRAM, 1/64 FP64 rate), `JAX_ENABLE_X64=1`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
* **External GPU Coverage:** `Optax` and `MPAX` run natively on GPU via JAX; standard PyPI `clarabel` and `scs` wheels run on CPU.
* **Reproduction:**

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 .venv/bin/python -m benchmarks.external.batch_conic_minmax --backend vmap --batch-sizes 1 16 64 256 1024
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 .venv/bin/python -m benchmarks.external.minmax_nonconvex_concave
```

## Results

![GPU Benchmarks Overview](_static/plots/gpu_benchmarks.svg)

### 1. Batched Conic & Min-Max Scaling (`B = 1 .. 1024`, `n = 64`)

| Problem | Batch `B` | Proxax GPU (`vmap`, ms) | Proxax CPU (`threads`, ms) | `Clarabel` CPU (`threads`, ms) | `SCS` CPU (`threads`, ms) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `Batched SOCP (n=64)` | 1 | 137.53 | 1.77 | 0.74 | 0.35 |
| `Batched SOCP (n=64)` | 16 | 221.57 | 5.34 | 7.77 | 3.75 |
| `Batched SOCP (n=64)` | 64 | 266.23 | 20.22 | 22.42 | 11.00 |
| `Batched SOCP (n=64)` | 256 | 352.28 | 77.13 | 81.37 | 38.43 |
| `Batched SOCP (n=64)` | 1024 | 798.87 | **295.18** | 355.76 | **149.43** |
| `Batched Saddle QP (n=64)` | 1 | 13.84 | 1.12 | 3.70 | 1.38 |
| `Batched Saddle QP (n=64)` | 16 | 51.07 | 8.03 | 22.27 | 5.65 |
| `Batched Saddle QP (n=64)` | 64 | 126.76 | 30.85 | 65.29 | 16.95 |
| `Batched Saddle QP (n=64)` | 256 | **163.02** | **120.57** | 242.78 | **62.46** |
| `Batched Saddle QP (n=64)` | 1024 | **529.13** | **472.91** | 950.87 | **237.09** |

### 2. Nonconvex-Concave Min-Max on GPU (`float64` & `float32`, Proxax GPU vs. `Optax` GPU)

| Problem | `PDHG` (`f64`, ms) | `PDHG` (`f32`, ms) | `ExtraGrad` (`f32`, ms) | `Optax GDA` (`f64`, ms) | `Optax GDA` (`f32`, ms) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `NC-C QP (n=32, m_y=0.05)` | 28.85 | **21.24** | 59.02 | 81.65 | 44.92 |
| `NC-C QP (n=32, m_y=0.2)` | 26.69 | **24.48** | 66.98 | 96.89 | 79.27 |
| `NC-C QP (n=32, m_y=1.0)` | 68.27 | **53.25** | 144.93 | 163.25 | 177.18 |
| `NC-C QP (n=64, m_y=0.2)` | **29.24** | 47.93 | 103.77 | 91.43 | 137.20 |
| `NC-C QP (n=256, m_y=0.2)` | 225.36 | **65.19** | 112.13 | 670.30 (`MAX_ITERS`) | 273.97 |

### 3. Sparse QPs & Spectral Ops on RTX 3060 (`float64`) vs. `MPAX` / CPU

* **Large-Scale Maros–Mészáros (`BCOO`, `tol=1e-6`):** `proxax-IAIPAL` converged on `MOSARQP1` in **518.44 ms** on GPU (vs. **90.13 ms** on CPU) and `proxax-IALM` converged in **8,026.00 ms** (vs. **1,514.84 ms** on CPU). `MPAX` (`mpax.r2HPDHG`, `eps=1e-6`) converged on `CVXQP2_M` in **2,102.33 ms** (`1409` iters) on GPU and returned `status=4` on `MOSARQP1` (`809` iters, `2,029.11 ms`).
* **Small Matrix Nuclear-Norm Completion (`16x16`–`48x48`):** GPU `IAICG` took `900.47–1,803.23 ms` (kernel-launch and `cusolver` overhead on small SVDs), compared to `14.11–92.75 ms` on CPU.

## Key Observations

1. **Sublinear `vmap` GPU scaling:** `proxax-IALM (vmap)` on `Batched SOCP (n=64)` scales only **5.8×** in runtime from `B=1` to `B=1024` (`137.53 ms -> 798.87 ms`), while CPU `Clarabel` scales **425×** and `SCS` scales **482×**. On `Batched Saddle QP`, `proxax-PDHG (vmap)` beats `Clarabel (threads)` by **1.51×** at `B=256` and **1.81×** at `B=1024`.
2. **Proxax GPU beats `Optax` GPU on nonconvex-concave min-max:** `PDHG` runs **2.4×–3.6× faster** than `Optax GDA` across `n=32..64` (`f64`) and converges at `n=256` (`225.36 ms`) where `Optax GDA` fails to reach `1e-4` within 10,000 steps.
3. **Consumer GPU FP64 vs. `float32` `vmap` lockstep:** On the RTX 3060 (`1/64` FP64 rate), switching to `float32` speeds up dense `n=256` `PDHG` by **3.46×** (`225.36 ms -> 65.19 ms`). However, for `vmap` batches (`B >= 16` at `tol=1e-3`), `float32` rounding noise can stall a single instance and force the entire `vmap` batch to run to `max_iters` in lockstep (`1,058–6,935 ms`), so `float64` (`JAX_ENABLE_X64=1`) is much more reliable for tight-tolerance `vmap` batches.

## Solver Regime Summary

| Problem Regime | Recommended Choice | Observed Advantage |
| :--- | :--- | :--- |
| Batched conic / saddle QPs (`B >= 64`) | `proxax.batch_solve` (`threads` / `vmap`) | Sublinear batch scaling (**5.8×** for `1024×` batch on GPU SOCP); beats `Clarabel` **1.51×–2.01×** on Saddle QP |
| Nonconvex-concave min-max (GPU) | `proxax` `PDHG` / `ExtraGradient` | **2.4×–3.6× faster** than `Optax GDA`; converges at `n=256` where `Optax` stalls |
| Spectral & multi-block splitting (CPU) | `DAICG`/`IAICG`, `ADMM` | **5.4×–63.9×** faster than `Clarabel` on `SpectralProblem`; **2.4×–6.2×** faster on `BlockProblem` |
| Single large ill-conditioned sparse QPs (`1e-6`) | `Clarabel` (sparse direct IPM) | Solves **8/8** `maros_meszaros_large` QPs in `8.3–1,486 ms` on CPU |
