# `PAdaptADMM` CPU/GPU & `jax.vmap` Scaling Study

This report compares **CPU sequential**, **CPU `jax.vmap`**, and **GPU `jax.vmap`**
execution modes for `px.solvers.PAdaptADMM` (`adaptive_tau=False`, `float32`)
on batches of $K$ nonconvex Box-constrained Distributed QP (DQP) instances.

## Setup

- **Hardware:** CPU + NVIDIA GeForce RTX 3060 (12 GiB)
- **Precision:** `float32` (consumer GeForce `float64` throughput is 1/64th of `float32`)
- **Problem:** $B = 2$ blocks, $x_i \in [-1, 1]^n$, $A_i \in \mathbb{R}^{m \times n}$
  with i.i.d. $\mathcal{N}(0, 1/n)$ entries, $P_i = Q_i D_i Q_i^\top$ with
  eigenvalues in $[0.5, 2.0]$ except one negative eigenvalue $-0.05$.
- **Solver:** `PAdaptADMM` with `adaptive_tau=False`, relative tolerances
  `rel_tol = 1e-2`, `gamma_init = 10.0`, `max_stages = 15`,
  `max_stage_iters = 500`, `max_inner_iters = 100`.

## Reproduction

```bash
uv run python experiments/padapt_vmap_gpu/run.py
```

Script: `experiments/padapt_vmap_gpu/run.py`

## Results

```{table}
:class: grouped-3-table
| $B$ | $(m, n)$ | $K$ | Variant | Conv | Avg Iters | Cold (s) | Warm (s) | ms / inst |
|---:|:---|---:|:---|---:|---:|---:|---:|---:|
| 2 | (16, 32) | 1 | `CPU sequential` | 1/1 | 43.0 | 1.177 | **0.001** | **1.37** |
| 2 | (16, 32) | 1 | `CPU vmap` | 1/1 | 43.0 | 1.775 | **0.001** | **0.67** |
| 2 | (16, 32) | 1 | `GPU vmap` | 1/1 | 43.0 | 1.028 | 0.213 | 212.76 |
| 2 | (16, 32) | 8 | `CPU sequential` | 7/8 | 730.5 | 0.325 | **0.257** | **32.07** |
| 2 | (16, 32) | 8 | `CPU vmap` | 7/8 | 730.5 | 4.638 | 1.416 | 176.94 |
| 2 | (16, 32) | 8 | `GPU vmap` | 7/8 | 730.5 | 135.395 | 130.155 | 16269.34 |
| 2 | (16, 32) | 32 | `CPU sequential` | 27/32 | 918.2 | 1.319 | **1.254** | **39.19** |
| 2 | (16, 32) | 32 | `CPU vmap` | 27/32 | 918.2 | 17.995 | 14.804 | 462.64 |
| 2 | (16, 32) | 32 | `GPU vmap` | 28/32 | 732.0 | 286.342 | 286.229 | 8944.65 |
| 2 | (64, 128) | 1 | `CPU sequential` | 1/1 | 44.0 | 2.024 | **0.009** | **8.54** |
| 2 | (64, 128) | 1 | `CPU vmap` | 1/1 | 44.0 | 2.604 | **0.009** | **8.53** |
| 2 | (64, 128) | 1 | `GPU vmap` | 1/1 | 44.0 | 1.383 | 0.366 | 366.48 |
| 2 | (64, 128) | 8 | `CPU sequential` | 7/8 | 668.5 | 2.207 | **2.138** | **267.22** |
| 2 | (64, 128) | 8 | `CPU vmap` | 7/8 | 668.5 | 28.000 | 14.398 | 1799.72 |
| 2 | (64, 128) | 8 | `GPU vmap` | 7/8 | 668.5 | 114.918 | 119.693 | 14961.66 |
| 2 | (64, 128) | 32 | `CPU sequential` | 28/32 | 669.4 | 9.722 | **9.675** | **302.34** |
| 2 | (64, 128) | 32 | `CPU vmap` | 28/32 | 669.4 | 224.854 | 223.522 | 6985.05 |
| 2 | (64, 128) | 32 | `GPU vmap` | 28/32 | 669.4 | 243.203 | 241.503 | 7546.98 |
```

## Key Observations

1. **Straggler effect in `jax.vmap(while_loop)`:**
   Because `PAdaptADMM` uses nested `jax.lax.while_loop`s, a batched `vmap`
   continues executing all $K$ instances until the slowest straggler in the
   batch terminates. In `CPU sequential`, ~88% of instances converge in ~44
   outer iterations while only the few non-converged stragglers run to the
   stage limit, making `CPU sequential` **11.8×–23× faster** than `CPU vmap`
   at $K = 32$.
2. **GPU arithmetic intensity vs. kernel-launch overhead:**
   For small matrices (`(16, 32)`, $K=8$), nested `PF.ACG` `while_loop`s on
   GPU are dominated by CUDA kernel-launch and host-sync latency (`130.2s` vs.
   `1.42s` on `CPU vmap`, **92× slower**). Increasing both matrix size to
   `(64, 128)` and batch size to $K = 32$ raises arithmetic intensity enough
   that `GPU vmap` (`241.5s`) reaches near-parity (**1.08×**) with `CPU vmap`
   (`223.5s`).
