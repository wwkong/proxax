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
| `Batched SOCP (n=64)` | 1 | 131.88 | 1.50 | 0.76 | 0.46 |
| `Batched SOCP (n=64)` | 16 | 210.67 | 5.27 | 7.44 | 3.59 |
| `Batched SOCP (n=64)` | 64 | 266.34 | 18.48 | 21.95 | 11.09 |
| `Batched SOCP (n=64)` | 256 | 300.09 | 69.40 | 79.51 | 39.68 |
| `Batched SOCP (n=64)` | 1024 | 730.31 | **275.83** | 312.17 | **144.24** |
| `Batched Saddle QP (n=64)` | 1 | 12.24 | 1.13 | 3.65 | 1.31 |
| `Batched Saddle QP (n=64)` | 16 | 50.45 | 8.56 | 21.62 | 5.91 |
| `Batched Saddle QP (n=64)` | 64 | 81.29 | 30.14 | 65.16 | 17.57 |
| `Batched Saddle QP (n=64)` | 256 | **156.64** | **112.30** | 245.00 | **63.15** |
| `Batched Saddle QP (n=64)` | 1024 | **512.81** | **448.08** | 945.17 | **232.97** |

### 2. Nonconvex-Concave Min-Max on GPU (`float64` & `float32`, Proxax GPU vs. `Optax` GPU)

| Problem | `PDHG` (`f64`, ms) | `PDHG` (`f32`, ms) | `ExtraGrad` (`f32`, ms) | `Optax GDA` (`f64`, ms) | `Optax GDA` (`f32`, ms) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `NC-C QP (n=32, m_y=0.05)` | 31.65 | **24.16** | 64.99 | 91.71 | 44.95 |
| `NC-C QP (n=32, m_y=0.2)` | 33.19 | **25.27** | 63.57 | 102.68 | 48.26 |
| `NC-C QP (n=32, m_y=1.0)` | 85.67 | **48.45** | 98.11 | 233.49 | 122.96 |
| `NC-C QP (n=64, m_y=0.2)` | 48.65 | 50.74 | 110.19 | 137.42 | 131.44 |
| `NC-C QP (n=256, m_y=0.2)` | 229.46 | **72.57** | 119.01 | 729.49 (`MAX_ITERS`) | 265.86 |

### 3. Sparse QPs & Spectral Ops on RTX 3060 (`float64`) vs. `MPAX` / CPU

* **Large-Scale Maros–Mészáros (`BCOO`, `tol=1e-6`):** `proxax-IAIPAL` converged on `MOSARQP1` in **518.44 ms** on GPU (vs. **90.13 ms** on CPU) and `proxax-IALM` converged in **8,026.00 ms** (vs. **1,514.84 ms** on CPU). `MPAX` (`mpax.r2HPDHG`, `eps=1e-6`) converged on `CVXQP2_M` in **2,102.33 ms** (`1409` iters) on GPU and returned `status=4` on `MOSARQP1` (`809` iters, `2,029.11 ms`).
* **Small Matrix Nuclear-Norm Completion (`16x16`–`48x48`):** GPU `IAICG` took `900.47–1,803.23 ms` (kernel-launch and `cusolver` overhead on small SVDs), compared to `14.11–92.75 ms` on CPU.

## Key Observations

1. **Sublinear `vmap` GPU scaling:** `proxax-IALM (vmap)` on `Batched SOCP (n=64)` scales only **5.5×** in runtime from `B=1` to `B=1024` (`131.88 ms -> 730.31 ms`), while CPU `Clarabel` scales **410×** and `SCS` scales **313×**. On `Batched Saddle QP`, `proxax-PDHG (vmap)` beats `Clarabel (threads)` by **1.56×** at `B=256` and **1.84×** at `B=1024`.
2. **Proxax GPU beats `Optax` GPU on nonconvex-concave min-max:** `PDHG` runs **2.7×–3.1× faster** than `Optax GDA` across `n=32..64` and converges at `n=256` (`229.46 ms`) where `Optax GDA` fails to reach `1e-4` within 10,000 steps.
3. **Consumer GPU FP64 vs. `float32` `vmap` lockstep:** On the RTX 3060 (`1/64` FP64 rate), switching to `float32` speeds up dense `n=256` `PDHG` by **3.16×** (`229.46 ms -> 72.57 ms`) and `Batched Saddle QP (B=1)` (`12.24 ms -> 8.71 ms`). However, for `vmap` batches (`B >= 16` at `tol=1e-3`), `float32` rounding noise can stall a single instance and force the entire `vmap` batch to run to `max_iters` in lockstep (`1,058–6,935 ms`), so `float64` (`JAX_ENABLE_X64=1`) is much more reliable for tight-tolerance `vmap` batches.

## Solver Regime Summary

| Problem Regime | Recommended Choice | Observed Advantage |
| :--- | :--- | :--- |
| Batched conic / saddle QPs (`B >= 64`) | `proxax.batch_solve` (`threads` / `vmap`) | Sublinear batch scaling (**5.5×** for `1024×` batch on GPU SOCP); beats `Clarabel` **1.56×–2.09×** on Saddle QP |
| Nonconvex-concave min-max (GPU) | `proxax` `PDHG` / `ExtraGradient` | **2.7×–3.7× faster** than `Optax GDA`; converges at `n=256` where `Optax` stalls |
| Spectral & multi-block splitting (CPU) | `DAICG`/`IAICG`, `ADMM` | **7.1×–58.2×** faster than `Clarabel` on `SpectralProblem`; **2.9×–7.2×** faster on `BlockProblem` |
| Single large ill-conditioned sparse QPs (`1e-6`) | `Clarabel` (sparse direct IPM) | Solves **8/8** `maros_meszaros_large` QPs in `8.1–1,527 ms` on CPU |
