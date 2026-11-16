# `QPALM` with `qdldl_ffi`, `BCOO` Sparsity, and Built-in Ruiz Equilibration

Benchmarking `QPALM` on the 23 medium-scale Maros–Meszaros convex QPs (`n = 2..900`, `m = 1..1001`) across three solver configurations against `Clarabel` (`0.11.1`, Rust IPM baseline), quantifying the impact of:

1. **Custom C++ `qdldl_ffi` KKT solver** (`src/proxax/solvers/constrained/_sparse_ldl_ffi.py`):
   - Hybrid sparse effective Hessian ($K_{\text{eff}}$ for sparse rows with $\text{row\_nnz} \le 32$, 2×2 KKT border for dense rows/active constraints) with 2-step iterative refinement.
   - Greedy Minimum Degree (GMD) fill-reducing ordering, `m_dense > 0` `thread_local` symbolic factorization caching, `O(n + m)` `SpotHashBuffers` workspace reuse, and zero-copy `Li` indexing in `SolveNumericWithSymbolic`.
2. **Sparse `BCOO` matrix support & incremental inner-loop tracking** (`src/proxax/solvers/constrained/qpalm.py`):
   - Direct `qdldl_kkt_solve_bcoo` dispatch avoiding dense $O((n+m)^2)$ materialization.
   - Incremental `(g_fx, ax_cur)` tracking in `_inner_body_q` to eliminate redundant matrix-vector products along line-search / Newton steps.
3. **Built-in Modified Ruiz Equilibration (`equilibrate=True`)** (`src/proxax/solvers/constrained/qpalm.py`):
   - Scales `(h_f_const, q_lin, a_mat)` internally while checking unscaled outer KKT residuals (`e_c_orig`, `e_x_orig`, `v_orig`) and unscaled inner gradient stopping (`g_chk = g_cur / (cost_scale * d_x)`).

## Setup

* **Precision & Device:** `float64` (`JAX_ENABLE_X64=1`), CPU.
* **Tolerances:** `opt_tol = 1e-6`, `feas_tol = 1e-6`.
* **Time Limit:** `time_limit = 10.0` seconds per solve (`jit=True` wall-clock cap).
* **Timing:** 1 JIT warmup call (`compile_s`) followed by `repeats=3` timed calls (`run_s`), reported in milliseconds.
* **Shifted Geometric Mean:** $\text{SGM}(10) = \exp\bigl(\frac{1}{N}\sum_{i=1}^N \ln(\max(t_i, 0) + 10)\bigr) - 10\text{ ms}$ (using `10000 ms` for non-converged runs).

### Reproduction

```bash
JAX_ENABLE_X64=1 .venv/bin/python experiments/qpalm_qdldl_ffi/run.py
```

Raw JSON results are written to `experiments/qpalm_qdldl_ffi/results/results.json`.

## Results

### Summary (23 Maros–Meszaros Medium QPs)

| Solver | Solved | SGM(10) (ms) | Avg (ms) | Max (ms) | Avg Compile (s) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `Clarabel` (baseline) | 23 / 23 | 2.04 | 2.41 | 16.73 | — |
| `QPALM (dense)` | 22 / 23 | 45.88 | 774.57 | 10252.73 | 1.27 |
| `QPALM (qdldl_ffi)` | **23 / 23** | **3.16** | **3.70** | **19.66** | **0.37** |
| `QPALM (qdldl_ffi+Ruiz)` | **23 / 23** | 4.44 | 5.77 | 26.95 | 0.51 |

### Key Findings

* **Dense vs. `qdldl_ffi` (`BCOO`):** Switching from dense linear solves to `qdldl_ffi` + `BCOO` cuts `SGM(10)` by **14.5×** (`45.88 ms` → `3.16 ms`), average runtime by **209×** (`774.57 ms` → `3.70 ms`), and average JIT compile time by **3.4×** (`1.27 s` → `0.37 s`), while raising solve rate from `22/23` to `23/23`.
  * `GOULDQP2 (n=699, m=349)`: `5309.11 ms` → **`10.21 ms`** (**520×** faster), `27(2555)` → `5(214)` iters.
  * `PRIMALC8 (n=520, m=8)`: `>10 s` timeout → **`2.86 ms`**.
  * `PRIMALC1 (n=230, m=9)`: `905.57 ms` → **`5.61 ms`** (**161×** faster).
  * `PRIMAL3 (n=745, m=111)`: `682.67 ms` → **`19.66 ms`** (**34.7×** faster).
  * `MOSARQP2 (n=900, m=600)`: `248.71 ms` → **`9.89 ms`** (**25.1×** faster).
* **Impact of Built-in Ruiz Equilibration (`equilibrate=True`):**
  * Substantially reduces outer and inner iterations on ill-conditioned problems:
    * `PRIMALC1`: `7(215)` → **`3(6)`** (`5.61 ms` → **`2.44 ms`**)
    * `DUALC2`: `8(26)` → **`3(14)`** (`1.11 ms` → **`0.84 ms`**)
    * `DUALC5`: `8(18)` → **`2(6)`** (`1.33 ms` → **`0.64 ms`**)
    * `DUALC8`: `9(37)` → **`3(14)`** (`1.68 ms` → **`1.23 ms`**)
    * `PRIMALC5`: `6(18)` → **`3(8)`**, `PRIMALC8`: `7(16)` → **`3(7)`**
  * Because equilibration runs inside JAX on `(h_f_const, a_mat)` per solve, it adds `~1.3 ms` to `SGM(10)` (`4.44 ms` vs. `3.16 ms`) on already well-scaled sparse QPs.

### Full Results

| Problem | Solver | Status | Outer | Inner | `opt_res` | `feas_res` | Runtime (ms) | Compile (s) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `CVXQP1_S (n=100,m=50)` | `Clarabel` | `CONVERGED` | 8 | 0 | `3.46e-16` | `6.16e-15` | 0.60 | 0.00 |
| `CVXQP1_S (n=100,m=50)` | `QPALM (dense)` | `CONVERGED` | 8 | 22 | `1.39e-08` | `9.22e-07` | 11.05 | 0.56 |
| `CVXQP1_S (n=100,m=50)` | `QPALM (qdldl_ffi)` | `CONVERGED` | 8 | 21 | `5.50e-09` | `9.22e-07` | 1.64 | 0.39 |
| `CVXQP1_S (n=100,m=50)` | `QPALM (qdldl_ffi+Ruiz)` | `CONVERGED` | 8 | 17 | `2.12e-07` | `8.30e-08` | 1.50 | 0.50 |
| `CVXQP2_S (n=100,m=25)` | `Clarabel` | `CONVERGED` | 8 | 0 | `3.72e-16` | `5.58e-14` | 0.58 | 0.00 |
| `CVXQP2_S (n=100,m=25)` | `QPALM (dense)` | `CONVERGED` | 9 | 16 | `2.04e-10` | `3.36e-07` | 7.15 | 0.52 |
| `CVXQP2_S (n=100,m=25)` | `QPALM (qdldl_ffi)` | `CONVERGED` | 9 | 16 | `6.30e-11` | `3.36e-07` | 1.45 | 0.37 |
| `CVXQP2_S (n=100,m=25)` | `QPALM (qdldl_ffi+Ruiz)` | `CONVERGED` | 7 | 13 | `8.89e-11` | `1.41e-07` | 1.28 | 0.50 |
| `CVXQP3_S (n=100,m=75)` | `Clarabel` | `CONVERGED` | 9 | 0 | `2.85e-16` | `2.50e-11` | 0.85 | 0.00 |
| `CVXQP3_S (n=100,m=75)` | `QPALM (dense)` | `CONVERGED` | 10 | 27 | `2.58e-07` | `3.85e-07` | 10.44 | 0.53 |
| `CVXQP3_S (n=100,m=75)` | `QPALM (qdldl_ffi)` | `CONVERGED` | 10 | 24 | `2.99e-07` | `3.85e-07` | 1.81 | 0.39 |
| `CVXQP3_S (n=100,m=75)` | `QPALM (qdldl_ffi+Ruiz)` | `CONVERGED` | 9 | 25 | `2.14e-07` | `1.12e-07` | 2.12 | 0.50 |
| `DPKLO1 (n=133,m=77)` | `Clarabel` | `CONVERGED` | 0 | 0 | `6.00e-16` | `9.94e-16` | 0.21 | 0.00 |
| `DPKLO1 (n=133,m=77)` | `QPALM (dense)` | `CONVERGED` | 4 | 4 | `3.31e-12` | `1.73e-07` | 4.82 | 0.49 |
| `DPKLO1 (n=133,m=77)` | `QPALM (qdldl_ffi)` | `CONVERGED` | 4 | 4 | `6.26e-13` | `1.73e-07` | 0.79 | 0.34 |
| `DPKLO1 (n=133,m=77)` | `QPALM (qdldl_ffi+Ruiz)` | `CONVERGED` | 8 | 8 | `2.77e-12` | `8.30e-07` | 2.25 | 0.49 |
| `DUALC1 (n=9,m=215)` | `Clarabel` | `CONVERGED` | 10 | 0 | `1.10e-09` | `6.29e-14` | 0.65 | 0.00 |
| `DUALC1 (n=9,m=215)` | `QPALM (dense)` | `CONVERGED` | 10 | 45 | `2.28e-08` | `6.00e-07` | 5.92 | 0.56 |
| `DUALC1 (n=9,m=215)` | `QPALM (qdldl_ffi)` | `CONVERGED` | 10 | 43 | `1.12e-08` | `6.00e-07` | 1.72 | 0.39 |
| `DUALC1 (n=9,m=215)` | `QPALM (qdldl_ffi+Ruiz)` | `CONVERGED` | 4 | 71 | `3.28e-07` | `1.13e-13` | 1.49 | 0.47 |
| `DUALC2 (n=7,m=229)` | `Clarabel` | `CONVERGED` | 10 | 0 | `3.06e-10` | `6.58e-15` | 0.52 | 0.00 |
| `DUALC2 (n=7,m=229)` | `QPALM (dense)` | `CONVERGED` | 8 | 26 | `4.01e-09` | `1.74e-07` | 2.71 | 0.51 |
| `DUALC2 (n=7,m=229)` | `QPALM (qdldl_ffi)` | `CONVERGED` | 8 | 26 | `3.84e-09` | `1.74e-07` | 1.11 | 0.35 |
| `DUALC2 (n=7,m=229)` | `QPALM (qdldl_ffi+Ruiz)` | `CONVERGED` | 3 | 14 | `2.91e-07` | `1.69e-13` | 0.84 | 0.46 |
| `DUALC5 (n=8,m=278)` | `Clarabel` | `CONVERGED` | 9 | 0 | `5.36e-09` | `9.06e-14` | 0.60 | 0.00 |
| `DUALC5 (n=8,m=278)` | `QPALM (dense)` | `CONVERGED` | 8 | 18 | `3.88e-10` | `1.45e-08` | 2.45 | 0.46 |
| `DUALC5 (n=8,m=278)` | `QPALM (qdldl_ffi)` | `CONVERGED` | 8 | 18 | `1.10e-10` | `1.45e-08` | 1.33 | 0.33 |
| `DUALC5 (n=8,m=278)` | `QPALM (qdldl_ffi+Ruiz)` | `CONVERGED` | 2 | 6 | `6.83e-07` | `1.30e-07` | 0.64 | 0.47 |
| `DUALC8 (n=8,m=503)` | `Clarabel` | `CONVERGED` | 9 | 0 | `2.16e-08` | `7.91e-13` | 1.13 | 0.00 |
| `DUALC8 (n=8,m=503)` | `QPALM (dense)` | `CONVERGED` | 9 | 36 | `2.35e-09` | `9.67e-07` | 7.98 | 0.48 |
| `DUALC8 (n=8,m=503)` | `QPALM (qdldl_ffi)` | `CONVERGED` | 9 | 37 | `2.18e-09` | `9.67e-07` | 1.68 | 0.33 |
| `DUALC8 (n=8,m=503)` | `QPALM (qdldl_ffi+Ruiz)` | `CONVERGED` | 3 | 14 | `5.99e-08` | `9.44e-10` | 1.23 | 0.45 |
| `GENHS28 (n=10,m=8)` | `Clarabel` | `CONVERGED` | 0 | 0 | `4.96e-16` | `0.00e+00` | 0.01 | 0.00 |
| `GENHS28 (n=10,m=8)` | `QPALM (dense)` | `CONVERGED` | 4 | 4 | `2.62e-14` | `5.41e-08` | 0.25 | 0.32 |
| `GENHS28 (n=10,m=8)` | `QPALM (qdldl_ffi)` | `CONVERGED` | 4 | 4 | `1.51e-14` | `5.41e-08` | 0.63 | 0.29 |
| `GENHS28 (n=10,m=8)` | `QPALM (qdldl_ffi+Ruiz)` | `CONVERGED` | 5 | 5 | `3.71e-14` | `1.45e-07` | 0.54 | 0.40 |
| `GOULDQP2 (n=699,m=349)` | `Clarabel` | `CONVERGED` | 10 | 0 | `1.66e-16` | `9.27e-10` | 2.73 | 0.00 |
| `GOULDQP2 (n=699,m=349)` | `QPALM (dense)` | `CONVERGED` | 27 | 2555 | `4.14e-07` | `2.12e-08` | 5309.11 | 5.88 |
| `GOULDQP2 (n=699,m=349)` | `QPALM (qdldl_ffi)` | `CONVERGED` | 5 | 214 | `3.97e-12` | `8.56e-08` | 10.21 | 0.40 |
| `GOULDQP2 (n=699,m=349)` | `QPALM (qdldl_ffi+Ruiz)` | `CONVERGED` | 5 | 215 | `4.02e-12` | `8.41e-08` | 19.82 | 0.52 |
| `GOULDQP3 (n=699,m=349)` | `Clarabel` | `CONVERGED` | 6 | 0 | `1.10e-15` | `1.47e-07` | 1.98 | 0.00 |
| `GOULDQP3 (n=699,m=349)` | `QPALM (dense)` | `CONVERGED` | 5 | 10 | `4.70e-12` | `1.85e-07` | 7.90 | 0.47 |
| `GOULDQP3 (n=699,m=349)` | `QPALM (qdldl_ffi)` | `CONVERGED` | 5 | 10 | `4.38e-13` | `1.85e-07` | 7.29 | 0.38 |
| `GOULDQP3 (n=699,m=349)` | `QPALM (qdldl_ffi+Ruiz)` | `CONVERGED` | 3 | 10 | `1.38e-11` | `5.45e-08` | 12.85 | 0.56 |
| `HS35MOD (n=3,m=1)` | `Clarabel` | `CONVERGED` | 9 | 0 | `2.58e-16` | `8.13e-17` | 0.02 | 0.00 |
| `HS35MOD (n=3,m=1)` | `QPALM (dense)` | `CONVERGED` | 3 | 6 | `1.49e-12` | `5.67e-07` | 0.21 | 0.37 |
| `HS35MOD (n=3,m=1)` | `QPALM (qdldl_ffi)` | `CONVERGED` | 3 | 6 | `1.48e-12` | `5.67e-07` | 0.26 | 0.28 |
| `HS35MOD (n=3,m=1)` | `QPALM (qdldl_ffi+Ruiz)` | `CONVERGED` | 3 | 7 | `1.21e-12` | `3.61e-09` | 0.26 | 0.38 |
| `KSIP (n=20,m=1001)` | `Clarabel` | `CONVERGED` | 13 | 0 | `1.39e-07` | `2.67e-09` | 7.72 | 0.00 |
| `KSIP (n=20,m=1001)` | `QPALM (dense)` | `CONVERGED` | 7 | 24 | `3.91e-10` | `1.88e-07` | 42.13 | 0.58 |
| `KSIP (n=20,m=1001)` | `QPALM (qdldl_ffi)` | `CONVERGED` | 7 | 24 | `1.42e-10` | `1.88e-07` | 3.61 | 0.44 |
| `KSIP (n=20,m=1001)` | `QPALM (qdldl_ffi+Ruiz)` | `CONVERGED` | 7 | 24 | `1.42e-10` | `1.88e-07` | 4.71 | 0.63 |
| `MOSARQP2 (n=900,m=600)` | `Clarabel` | `CONVERGED` | 8 | 0 | `6.50e-13` | `2.41e-13` | 5.36 | 0.00 |
| `MOSARQP2 (n=900,m=600)` | `QPALM (dense)` | `CONVERGED` | 6 | 62 | `3.26e-07` | `8.77e-09` | 248.71 | 0.75 |
| `MOSARQP2 (n=900,m=600)` | `QPALM (qdldl_ffi)` | `CONVERGED` | 5 | 14 | `4.59e-13` | `7.86e-08` | 9.89 | 0.39 |
| `MOSARQP2 (n=900,m=600)` | `QPALM (qdldl_ffi+Ruiz)` | `CONVERGED` | 3 | 18 | `2.77e-11` | `8.34e-08` | 22.31 | 0.53 |
| `PRIMAL1 (n=325,m=85)` | `Clarabel` | `CONVERGED` | 8 | 0 | `2.62e-07` | `2.25e-08` | 3.61 | 0.00 |
| `PRIMAL1 (n=325,m=85)` | `QPALM (dense)` | `CONVERGED` | 2 | 13 | `1.28e-07` | `6.36e-07` | 57.35 | 0.58 |
| `PRIMAL1 (n=325,m=85)` | `QPALM (qdldl_ffi)` | `CONVERGED` | 2 | 13 | `1.84e-11` | `6.36e-07` | 2.77 | 0.45 |
| `PRIMAL1 (n=325,m=85)` | `QPALM (qdldl_ffi+Ruiz)` | `CONVERGED` | 2 | 13 | `1.84e-11` | `6.36e-07` | 4.93 | 0.53 |
| `PRIMAL2 (n=649,m=96)` | `Clarabel` | `CONVERGED` | 7 | 0 | `6.77e-07` | `6.12e-08` | 5.48 | 0.00 |
| `PRIMAL2 (n=649,m=96)` | `QPALM (dense)` | `CONVERGED` | 3 | 18 | `4.80e-08` | `3.87e-10` | 159.66 | 0.69 |
| `PRIMAL2 (n=649,m=96)` | `QPALM (qdldl_ffi)` | `CONVERGED` | 2 | 17 | `8.88e-12` | `9.60e-08` | 7.04 | 0.41 |
| `PRIMAL2 (n=649,m=96)` | `QPALM (qdldl_ffi+Ruiz)` | `CONVERGED` | 2 | 17 | `8.88e-12` | `9.60e-08` | 13.58 | 0.57 |
| `PRIMAL3 (n=745,m=111)` | `Clarabel` | `CONVERGED` | 8 | 0 | `1.52e-08` | `1.30e-09` | 16.73 | 0.00 |
| `PRIMAL3 (n=745,m=111)` | `QPALM (dense)` | `CONVERGED` | 2 | 35 | `1.33e-07` | `1.47e-07` | 682.67 | 1.29 |
| `PRIMAL3 (n=745,m=111)` | `QPALM (qdldl_ffi)` | `CONVERGED` | 2 | 35 | `7.89e-12` | `1.47e-07` | 19.66 | 0.46 |
| `PRIMAL3 (n=745,m=111)` | `QPALM (qdldl_ffi+Ruiz)` | `CONVERGED` | 2 | 35 | `7.89e-12` | `1.47e-07` | 26.95 | 0.75 |
| `PRIMALC1 (n=230,m=9)` | `Clarabel` | `CONVERGED` | 16 | 0 | `1.58e-14` | `3.14e-16` | 1.35 | 0.00 |
| `PRIMALC1 (n=230,m=9)` | `QPALM (dense)` | `CONVERGED` | 13 | 342 | `7.72e-07` | `1.82e-12` | 905.57 | 1.39 |
| `PRIMALC1 (n=230,m=9)` | `QPALM (qdldl_ffi)` | `CONVERGED` | 7 | 215 | `6.05e-07` | `5.29e-08` | 5.61 | 0.37 |
| `PRIMALC1 (n=230,m=9)` | `QPALM (qdldl_ffi+Ruiz)` | `CONVERGED` | 3 | 6 | `2.05e-08` | `4.89e-12` | 2.44 | 0.54 |
| `PRIMALC2 (n=231,m=7)` | `Clarabel` | `CONVERGED` | 14 | 0 | `2.86e-09` | `4.74e-12` | 1.08 | 0.00 |
| `PRIMALC2 (n=231,m=7)` | `QPALM (dense)` | `CONVERGED` | 5 | 17 | `4.22e-07` | `0.00e+00` | 30.57 | 0.55 |
| `PRIMALC2 (n=231,m=7)` | `QPALM (qdldl_ffi)` | `CONVERGED` | 4 | 10 | `3.84e-07` | `0.00e+00` | 1.00 | 0.40 |
| `PRIMALC2 (n=231,m=7)` | `QPALM (qdldl_ffi+Ruiz)` | `CONVERGED` | 3 | 6 | `2.72e-08` | `2.72e-11` | 2.40 | 0.53 |
| `PRIMALC5 (n=287,m=8)` | `Clarabel` | `CONVERGED` | 13 | 0 | `3.75e-15` | `2.55e-15` | 1.32 | 0.00 |
| `PRIMALC5 (n=287,m=8)` | `QPALM (dense)` | `CONVERGED` | 6 | 23 | `6.19e-08` | `7.94e-09` | 65.36 | 0.68 |
| `PRIMALC5 (n=287,m=8)` | `QPALM (qdldl_ffi)` | `CONVERGED` | 6 | 18 | `3.47e-08` | `7.94e-09` | 2.38 | 0.41 |
| `PRIMALC5 (n=287,m=8)` | `QPALM (qdldl_ffi+Ruiz)` | `CONVERGED` | 3 | 8 | `1.08e-10` | `8.95e-08` | 2.96 | 0.54 |
| `PRIMALC8 (n=520,m=8)` | `Clarabel` | `CONVERGED` | 11 | 0 | `1.03e-08` | `8.11e-11` | 2.88 | 0.00 |
| `PRIMALC8 (n=520,m=8)` | `QPALM (dense)` | `TIME_LIMIT_REACHED` | 23 | 2038 | `2.95e-04` | `3.07e-05` | 10252.73 | 10.85 |
| `PRIMALC8 (n=520,m=8)` | `QPALM (qdldl_ffi)` | `CONVERGED` | 7 | 16 | `4.32e-07` | `4.38e-09` | 2.86 | 0.39 |
| `PRIMALC8 (n=520,m=8)` | `QPALM (qdldl_ffi+Ruiz)` | `CONVERGED` | 3 | 7 | `2.90e-08` | `8.29e-10` | 6.80 | 0.53 |
| `S268 (n=5,m=5)` | `Clarabel` | `CONVERGED` | 9 | 0 | `1.26e-11` | `4.88e-16` | 0.03 | 0.00 |
| `S268 (n=5,m=5)` | `QPALM (dense)` | `CONVERGED` | 1 | 2 | `4.00e-07` | `0.00e+00` | 0.19 | 0.32 |
| `S268 (n=5,m=5)` | `QPALM (qdldl_ffi)` | `CONVERGED` | 1 | 2 | `4.00e-07` | `0.00e+00` | 0.24 | 0.26 |
| `S268 (n=5,m=5)` | `QPALM (qdldl_ffi+Ruiz)` | `CONVERGED` | 5 | 6 | `3.12e-07` | `1.75e-15` | 0.50 | 0.48 |
| `ZECEVIC2 (n=2,m=2)` | `Clarabel` | `CONVERGED` | 7 | 0 | `8.47e-11` | `3.66e-12` | 0.02 | 0.00 |
| `ZECEVIC2 (n=2,m=2)` | `QPALM (dense)` | `CONVERGED` | 2 | 6 | `1.00e-09` | `1.86e-10` | 0.21 | 0.42 |
| `ZECEVIC2 (n=2,m=2)` | `QPALM (qdldl_ffi)` | `CONVERGED` | 2 | 6 | `1.00e-09` | `1.86e-10` | 0.20 | 0.31 |
| `ZECEVIC2 (n=2,m=2)` | `QPALM (qdldl_ffi+Ruiz)` | `CONVERGED` | 2 | 3 | `1.00e-09` | `1.81e-10` | 0.22 | 0.40 |
