# Closed-Loop Warm-Started MPC Rollout (`proxax` vs. `Clarabel` & `alpaqa`)

CPU (`float64`) comparison of a **20-step closed-loop receding-horizon MPC** controller on a coupled 6-mass oscillator ($n_x=12$, $n_u=4$) across horizons $N \in \{10, 20\}$ and batch sizes $B \in \{1, 16\}$.

Unlike single-shot static benchmarks, `proxax` compiles the **entire 20-step closed-loop simulation** (`jax.lax.scan`) into one XLA graph (`jit(vmap(lax.scan))`), warm-starting each step by shifting the previous control sequence $z^{(t)} = (u_1^{(t-1)}, \dots, u_{N-1}^{(t-1)}, u_{N-1}^{(t-1)})$. `Clarabel` and `alpaqa` run an explicit Python control loop (with `ThreadPoolExecutor` across $B=16$ scenarios for `Clarabel`).

## Problem Classes

1. **Box-Constrained Condensed MPC (`mpc_box`)** —
   $$\min_{|u_k| \le 0.5} \sum_{k=0}^{N-1} \tfrac{1}{2}(x_k^\top Q x_k + u_k^\top R u_k) + \tfrac{1}{2} x_N^\top Q_f x_N$$
   Solved via condensed input-space QP with `proxax-PANOC` (cold & warm `lax.scan`), `alpaqa-PANOC` (warm, $B=1$), and `Clarabel`.
2. **State + Input Constrained Condensed MPC (`mpc_state_con`)** —
   Adds state bounds $|x_k|_\infty \le 1.5$ across the prediction horizon ($|u_k| \le 0.8$). Solved with `proxax-IAIPAL` (cold & warm `lax.scan`) and `Clarabel`.

## Setup

* **Precision & Device:** `float64` (`JAX_ENABLE_X64=1`), CPU.
* **Tolerances:** `tol = 1e-6`.
* **Timing:** Total 20-step closed-loop runtime (minimum over `repeats = 3` after warmup).
* **Reproduction:**

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=. .venv/bin/python benchmarks/external/mpc_rollout.py --tol 1e-6
```

## Results

### Horizon & Batch Scaling

![Closed-Loop MPC Rollout Scaling](_static/plots/mpc_rollout_scaling.svg)

### Dolan–Moré Runtime Profile

![Closed-Loop MPC Dolan-Moré Profile](_static/plots/mpc_rollout_profile.svg)

### Full Results (`tol = 1e-6`, 20 Closed-Loop Steps)

| Suite | Problem | Solver | $N$ | $B$ | Converged | Mean Cost | Compile (s) | Run (ms) | Speedup vs. `Clarabel` |
| :--- | :--- | :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: |
| `mpc_box` | `mpc_box_N10_B1` | `proxax-PANOC(cold)` | 10 | 1 | `True` | 1.9821 | 1.10 | 0.36 | 16.0× |
| `mpc_box` | `mpc_box_N10_B1` | `proxax-PANOC(warm)` | 10 | 1 | `True` | 1.9821 | 1.16 | **0.26** | **22.1×** |
| `mpc_box` | `mpc_box_N10_B1` | `alpaqa-PANOC(warm)` | 10 | 1 | `True` | 1.9821 | 0.00 | 1.66 | 3.5× |
| `mpc_box` | `mpc_box_N10_B1` | `Clarabel` | 10 | 1 | `True` | 1.9821 | 0.01 | 5.75 | 1.0× |
| `mpc_state_con` | `mpc_state_N10_B1` | `proxax-IAIPAL(cold)` | 10 | 1 | `True` | 1.9552 | 2.25 | 10.76 | 3.7× |
| `mpc_state_con` | `mpc_state_N10_B1` | `proxax-IAIPAL(warm)` | 10 | 1 | `True` | 1.9552 | 2.49 | **7.88** | **5.1×** |
| `mpc_state_con` | `mpc_state_N10_B1` | `Clarabel` | 10 | 1 | `True` | 1.9552 | 0.04 | 39.97 | 1.0× |
| `mpc_box` | `mpc_box_N20_B1` | `proxax-PANOC(cold)` | 20 | 1 | `True` | 1.9862 | 1.33 | 1.30 | 17.2× |
| `mpc_box` | `mpc_box_N20_B1` | `proxax-PANOC(warm)` | 20 | 1 | `True` | 1.9862 | 1.43 | **1.04** | **21.5×** |
| `mpc_box` | `mpc_box_N20_B1` | `alpaqa-PANOC(warm)` | 20 | 1 | `True` | 1.9862 | 0.00 | 3.25 | 6.9× |
| `mpc_box` | `mpc_box_N20_B1` | `Clarabel` | 20 | 1 | `True` | 1.9862 | 0.03 | 22.35 | 1.0× |
| `mpc_state_con` | `mpc_state_N20_B1` | `proxax-IAIPAL(cold)` | 20 | 1 | `True` | 1.9593 | 2.57 | 70.44 | 2.9× |
| `mpc_state_con` | `mpc_state_N20_B1` | `proxax-IAIPAL(warm)` | 20 | 1 | `True` | 1.9593 | 2.83 | **54.98** | **3.8×** |
| `mpc_state_con` | `mpc_state_N20_B1` | `Clarabel` | 20 | 1 | `True` | 1.9593 | 0.23 | 207.44 | 1.0× |
| `mpc_box` | `mpc_box_N10_B16` | `proxax-PANOC(cold)` | 10 | 16 | `True` | 1.7527 | 0.43 | 7.55 | 7.4× |
| `mpc_box` | `mpc_box_N10_B16` | `proxax-PANOC(warm)` | 10 | 16 | `True` | 1.7527 | 0.44 | **6.39** | **8.8×** |
| `mpc_box` | `mpc_box_N10_B16` | `Clarabel` | 10 | 16 | `True` | 1.7527 | 0.06 | 56.11 | 1.0× |
| `mpc_state_con` | `mpc_state_N10_B16` | `proxax-IAIPAL(cold)` | 10 | 16 | `True` | 1.7465 | 0.96 | 267.95 | 0.81× |
| `mpc_state_con` | `mpc_state_N10_B16` | `proxax-IAIPAL(warm)` | 10 | 16 | `True` | 1.7465 | 1.02 | **212.49** | **1.02×** |
| `mpc_state_con` | `mpc_state_N10_B16` | `Clarabel` | 10 | 16 | `True` | 1.7465 | 0.23 | 215.75 | 1.0× |
| `mpc_box` | `mpc_box_N20_B16` | `proxax-PANOC(cold)` | 20 | 16 | `True` | 1.7652 | 0.46 | 41.58 | 3.5× |
| `mpc_box` | `mpc_box_N20_B16` | `proxax-PANOC(warm)` | 20 | 16 | `True` | 1.7652 | 0.44 | **30.59** | **4.8×** |
| `mpc_box` | `mpc_box_N20_B16` | `Clarabel` | 20 | 16 | `True` | 1.7652 | 0.18 | 146.30 | 1.0× |
| `mpc_state_con` | `mpc_state_N20_B16` | `proxax-IAIPAL(cold)` | 20 | 16 | `True` | 1.7592 | 2.50 | 1805.12 | 0.46× |
| `mpc_state_con` | `mpc_state_N20_B16` | `proxax-IAIPAL(warm)` | 20 | 16 | `True` | 1.7592 | 2.48 | 1477.95 | 0.56× |
| `mpc_state_con` | `mpc_state_N20_B16` | `Clarabel` | 20 | 16 | `True` | 1.7592 | 0.84 | **825.60** | 1.0× |

## Key Takeaways

1. **Compiled `lax.scan` eliminates Python loop overhead:** For box-constrained MPC at $B=1$, `proxax-PANOC(warm)` executes all 20 closed-loop steps in **0.26 ms** ($N=10$, **13 µs/step**) and **1.04 ms** ($N=20$, **52 µs/step**) — **3.1×–6.4× faster** than warm-started `alpaqa-PANOC` and **21×–22× faster** than `Clarabel`.
2. **Warm-starting consistently saves 18%–28%:** Shifting the previous control sequence cuts closed-loop runtime across every single horizon and batch size (e.g., `70.44 ms` $\to$ `54.98 ms` on `mpc_state_N20_B1`, and `41.58 ms` $\to$ `30.59 ms` on `mpc_box_N20_B16`).
3. **Batched `vmap(lax.scan)` vs. threaded `Clarabel` on state-constrained MPC:** At $B=16, N=20$, `vmap` forces all 16 rollouts through lockstep `IAIPAL` iterations (`1477.95 ms`), whereas `Clarabel` runs 16 independent rollouts in parallel across CPU cores (`825.60 ms`). Using `ThreadPoolExecutor` over per-scenario `jit(lax.scan)` rollouts would remove that lockstep penalty on CPU.
