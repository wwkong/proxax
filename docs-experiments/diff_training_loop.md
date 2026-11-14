# End-to-End Differentiable Training Loop (`proxax` vs. `cvxpylayers`)

CPU (`float64`) comparison of a **50-step Adam training loop** learning linear cost parameters $\theta \in \mathbb{R}^n$ through a batched optimization layer across $B \in \{16, 64\}$ (`tol = 1e-6`).

Because `proxax` is fully XLA-traceable, `proxax(jit-scan)` compiles the **entire 50-step loop** (forward solve, implicit VJP, and Adam parameter updates via `jax.lax.scan`) into a single compiled XLA call. Standard `cvxpylayers` (`diffcp` backend) is not `jax.jit`-compatible, so each training step returns to eager Python.

## Problem Classes

1. **Batched QP Layer Training (`qp_train`)** —
   $$\min_{\theta} \frac{1}{2B}\sum_{i=1}^B \left\|x^*(q_i + \theta) - x_{\text{target}, i}\right\|_2^2, \qquad x^*(q) = \arg\min_{x \ge 0,\; Ax \le b} \tfrac{1}{2}x^\top P x + q^\top x \qquad (n=50,\; m=25)$$
2. **Batched Simplex Layer Training (`simplex_train`)** —
   $$\min_{\theta} \frac{1}{2B}\sum_{i=1}^B \left\|x^*(\mu_i + \theta) - x_{\text{target}, i}\right\|_2^2, \qquad x^*(\mu) = \arg\min_{x \in \Delta^n} \tfrac{1}{2}x^\top P x - \mu^\top x \qquad (n=100)$$

## Setup

* **Precision & Device:** `float64` (`JAX_ENABLE_X64=1`), CPU.
* **Optimizer:** 50 Adam steps (`lr = 0.05`, $\beta_1=0.9$, $\beta_2=0.999$).
* **Tolerances:** `tol = 1e-6` (forward & implicit linear solve; `eps = eps_abs = eps_rel = 1e-6` for `cvxpylayers`).
* **Timing:** Minimum over `repeats = 2` after warmup (`Compile (s)` reports first-call XLA compile + warmup or `CvxpyLayer` setup + first run).
* **Reproduction:**

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=. .venv/bin/python benchmarks/external/diff_training_loop.py --tol 1e-6
```

## Results

### 50-Step Training Scaling

![Proxax vs cvxpylayers 50-Step Training Scaling](_static/plots/diff_training_loop_scaling.svg)

### Dolan–Moré Runtime Profile

![Proxax vs cvxpylayers Training Dolan-Moré Profile](_static/plots/diff_training_loop_profile.svg)

### Full Results (`tol = 1e-6`, 50 Adam Steps)

| Suite | Problem | Solver | $B$ | Init Loss | Final Loss | Compile (s) | 50-Step Train (ms) | Speedup vs. `cvxpylayers` |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `qp_train` | `qp_train_n50_B16` | `proxax(jit-scan)` | 16 | 6.28084 | 1.21806 | 3.00 | **1167.81** | **2.81×** |
| `qp_train` | `qp_train_n50_B16` | `cvxpylayers` | 16 | 6.28084 | 1.21806 | 4.11 | 3278.00 | 1.0× |
| `simplex_train` | `simplex_train_n100_B16` | `proxax(jit-scan)` | 16 | 0.20976 | 0.14787 | 1.52 | **248.91** | **15.57×** |
| `simplex_train` | `simplex_train_n100_B16` | `cvxpylayers` | 16 | 0.20976 | 0.10458 | 4.67 | 3874.90 | 1.0× |
| `qp_train` | `qp_train_n50_B64` | `proxax(jit-scan)` | 64 | 6.47052 | 1.38402 | 7.61 | **5799.51** | **1.76×** |
| `qp_train` | `qp_train_n50_B64` | `cvxpylayers` | 64 | 6.47052 | 1.38402 | 11.00 | 10186.31 | 1.0× |
| `simplex_train` | `simplex_train_n100_B64` | `proxax(jit-scan)` | 64 | 0.20508 | 0.13569 | 2.35 | **1197.68** | **10.39×** |
| `simplex_train` | `simplex_train_n100_B64` | `cvxpylayers` | 64 | 0.20508 | 0.13178 | 13.04 | 12438.61 | 1.0× |

## Key Takeaways

1. **Whole-loop XLA compilation:** Fusing 50 Adam steps (`forward + implicit VJP + update`) in `jax.lax.scan` runs **10.4×–15.6× faster** than `cvxpylayers` on the simplex layer (`248.91 ms` vs. `3874.90 ms` at $B=16$, or **5.0 ms/step**) and **1.8×–2.8× faster** on the linearly constrained QP layer (`1167.81 ms` vs. `3278.00 ms` at $B=16$).
2. **Lower total warmup + first-epoch cost:** Even including XLA compilation (`1.52 s–7.61 s`), `proxax(jit-scan)` finishes its first 50-step epoch faster than `cvxpylayers`'s first-run setup (`4.11 s–13.04 s`).
3. **Training-loop note:** `qp_train` achieves identical initial and final losses (`6.28084` $\to$ `1.21806` at $B=16$), while on `simplex_train` the GMRES-based implicit VJP (`linear_tol=1e-6`) follows a slightly different trajectory than `diffcp`'s direct KKT solve (`0.14787` vs. `0.10458` at $B=16$).
