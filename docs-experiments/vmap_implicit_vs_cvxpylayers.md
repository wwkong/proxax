# Batched Differentiable Layers (`proxax` vs. `cvxpylayers`)

CPU (`float64`) comparison of `proxax` (`jax.vmap` + `px.implicit_solve`) against `cvxpylayers` (`1.2.0`, `CvxpyLayer` via `diffcp`) for end-to-end **forward solve + backward VJP** (`jax.value_and_grad`) across batch sizes $B \in \{1, 16, 64, 256\}$.

## Problem Classes

1. **Batched Linearly-Constrained QP Layer (`qp_layer`)** —
   $$\min_{x \ge 0} \tfrac{1}{2} x^\top P x + q_i^\top x \quad \text{s.t.} \quad A x \le b \qquad (n=50, \; m=25)$$
   with loss $\mathcal{L}(q_1, \dots, q_B) = \tfrac{1}{2}\sum_{i=1}^B \|x^*(q_i)\|_2^2$.
2. **Batched Simplex Layer (`simplex_layer`)** —
   $$\min_{x \in \Delta^{100}} \tfrac{1}{2} x^\top P x - \mu_i^\top x \qquad (n=100)$$
   with loss $\mathcal{L}(\mu_1, \dots, \mu_B) = \tfrac{1}{2}\sum_{i=1}^B \|x^*(\mu_i) - \tfrac{1}{2}\mathbf{1}\|_2^2$.
3. **Batched $\ell_2$-Ball Layer (`l2ball_layer`)** —
   $$\min_{\|x\|_2 \le 1} \tfrac{1}{2} x^\top P x + q_i^\top x \qquad (n=100)$$
   with loss $\mathcal{L}(q_1, \dots, q_B) = \tfrac{1}{2}\sum_{i=1}^B \|x^*(q_i) - \tfrac{1}{2}\mathbf{1}\|_2^2$. `proxax` projects directly with `px.prox.L2Ball(1.0)`, whereas `cvxpylayers` lifts $\|x\|_2 \le 1$ into a second-order cone in `diffcp`.

## Setup

* **Precision & Device:** `float64` (`JAX_ENABLE_X64=1`), CPU.
* **Tolerances:** `tol = 1e-6` (both forward optimality/feasibility and implicit linear solve tolerance; `solver_args={"eps": 1e-6, "eps_abs": 1e-6, "eps_rel": 1e-6}` for `cvxpylayers`).
* **Timing:** Steady-state minimum over `repeats = 3` after warmup (first call reported separately as `Compile (s)`).
* **Reproduction:**

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=. .venv/bin/python benchmarks/external/vmap_implicit_vs_cvxpylayers.py --tol 1e-6
```

## Results

### Batch Scaling (Forward + Backward VJP)

![Proxax vs cvxpylayers Batch Scaling](_static/plots/vmap_implicit_vs_cvxpylayers_scaling.svg)

### Full Results (`tol = 1e-6`)

| Suite | Problem | Solver | $B$ | Loss | Rel Grad Err | Compile (s) | Run (ms) | Speedup vs. `cvxpylayers` |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `qp_layer` | `qp_layer_n50_B1` | `proxax-IALM` | 1 | 15.0505 | `6.47e-06` | 2.74 | **0.93** | **17.70×** |
| `qp_layer` | `qp_layer_n50_B1` | `cvxpylayers` | 1 | 15.0505 | `0.00e+00` | 0.75 | 16.46 | 1.00× |
| `qp_layer` | `qp_layer_n50_B16` | `proxax-IALM` | 16 | 233.9316 | `8.52e-06` | 1.38 | **37.34** | **2.02×** |
| `qp_layer` | `qp_layer_n50_B16` | `cvxpylayers` | 16 | 233.9315 | `0.00e+00` | 1.30 | 75.32 | 1.00× |
| `qp_layer` | `qp_layer_n50_B64` | `proxax-IALM` | 64 | 779.2278 | `1.04e-05` | 1.44 | 257.92 | 0.96× |
| `qp_layer` | `qp_layer_n50_B64` | `cvxpylayers` | 64 | 779.2281 | `0.00e+00` | 1.62 | **248.88** | 1.00× |
| `qp_layer` | `qp_layer_n50_B256` | `proxax-IALM` | 256 | 2520.8586 | `1.07e-05` | 2.55 | 1113.57 | 0.79× |
| `qp_layer` | `qp_layer_n50_B256` | `cvxpylayers` | 256 | 2520.8582 | `0.00e+00` | 2.48 | **881.14** | 1.00× |
| `simplex_layer` | `simplex_layer_n100_B1` | `proxax-PANOC` | 1 | 12.2180 | `8.51e-06` | 1.01 | **0.70** | **33.54×** |
| `simplex_layer` | `simplex_layer_n100_B1` | `cvxpylayers` | 1 | 12.2180 | `0.00e+00` | 0.66 | 23.48 | 1.00× |
| `simplex_layer` | `simplex_layer_n100_B16` | `proxax-PANOC` | 16 | 195.1466 | `3.86e-06` | 0.88 | **8.34** | **12.87×** |
| `simplex_layer` | `simplex_layer_n100_B16` | `cvxpylayers` | 16 | 195.1466 | `0.00e+00` | 1.05 | 107.31 | 1.00× |
| `simplex_layer` | `simplex_layer_n100_B64` | `proxax-PANOC` | 64 | 781.8064 | `5.51e-06` | 0.93 | **37.19** | **10.01×** |
| `simplex_layer` | `simplex_layer_n100_B64` | `cvxpylayers` | 64 | 781.8064 | `0.00e+00` | 1.63 | 372.45 | 1.00× |
| `simplex_layer` | `simplex_layer_n100_B256` | `proxax-PANOC` | 256 | 3124.3430 | `4.27e-06` | 1.11 | **177.25** | **8.07×** |
| `simplex_layer` | `simplex_layer_n100_B256` | `cvxpylayers` | 256 | 3124.3431 | `0.00e+00` | 2.65 | 1430.25 | 1.00× |
| `l2ball_layer` | `l2ball_layer_n100_B1` | `proxax-PANOC` | 1 | 13.3139 | `1.08e-06` | 1.71 | **0.24** | **81.00×** |
| `l2ball_layer` | `l2ball_layer_n100_B1` | `cvxpylayers` | 1 | 13.3139 | `0.00e+00` | 0.69 | 19.44 | 1.00× |
| `l2ball_layer` | `l2ball_layer_n100_B16` | `proxax-PANOC` | 16 | 209.6644 | `1.25e-06` | 0.72 | **2.13** | **48.62×** |
| `l2ball_layer` | `l2ball_layer_n100_B16` | `cvxpylayers` | 16 | 209.6644 | `0.00e+00` | 1.05 | 103.55 | 1.00× |
| `l2ball_layer` | `l2ball_layer_n100_B64` | `proxax-PANOC` | 64 | 829.8868 | `1.39e-06` | 0.70 | **10.50** | **31.85×** |
| `l2ball_layer` | `l2ball_layer_n100_B64` | `cvxpylayers` | 64 | 829.8868 | `0.00e+00` | 1.42 | 334.40 | 1.00× |
| `l2ball_layer` | `l2ball_layer_n100_B256` | `proxax-PANOC` | 256 | 3310.2253 | `1.09e-06` | 0.81 | **34.74** | **38.33×** |
| `l2ball_layer` | `l2ball_layer_n100_B256` | `cvxpylayers` | 256 | 3310.2253 | `0.00e+00` | 2.49 | 1331.42 | 1.00× |

## Key Takeaways

* **$\ell_2$-Ball & Simplex layers (`proxax-PANOC`):** End-to-end XLA batching (`jax.vmap` + `px.implicit_solve`) beats `cvxpylayers` by **31.85–81.00×** on `l2ball_layer` (`34.74 ms` vs. `1331.42 ms` at $B=256$) and **8.07–33.54×** on `simplex_layer` (`177.25 ms` vs. `1430.25 ms` at $B=256$).
* **Linearly-constrained QP layers (`proxax-IALM`):** `proxax-IALM` is **17.70× faster** at $B=1$ (`0.93 ms` vs. `16.46 ms`), **2.02× faster** at $B=16$ (`37.34 ms` vs. `75.32 ms`), and near parity at $B=64$ (`257.92 ms` vs. `248.88 ms`) on CPU, where `jax.vmap` runs `lax.while_loop` in lockstep up to the worst-case iteration count in the batch (`max_outer=21, max_inner=1817` vs. `median_inner=755` at $B=256$).
* **Verified gradient agreement:** Relative $\ell_2$ gradient discrepancy against `cvxpylayers` stays between `1.08e-06` and `1.07e-05` across all 12 cases.
