# Multi-Block Nonconvex Splitting: `DPADMM` vs. `PAdaptADMM` (Kong & Monteiro, SIOPT 2024)

Small-scale sequential-CPU replication of **Kong & Monteiro (SIOPT 2024)**, *Global complexity bound of a proximal ADMM for linearly-constrained nonseperable nonconvex composite programming* (`DPADMM`), compared against **Maia et al. (2026)** (`PAdaptADMM`) and fixed-penalty `ADMM`.

## Setup & Reproduction

All runs execute sequentially on CPU via [`experiments/kong2024_dpadmm/run.py`](https://github.com/wwkong/proxax/blob/main/experiments/kong2024_dpadmm/run.py) (`opt_tol = 1e-3`, `feas_tol = 1e-3`), sweeping the third-block coupling matrix rank $r \in \{1, 2, 4, 6\}$ ($m=6$, so $r < 6$ is strictly rank-deficient):

```bash
JAX_PLATFORMS=cpu uv run python experiments/kong2024_dpadmm/run.py
```

## Rank-Sensitivity Plot

![Rank-Free Convergence of DPADMM](_static/plots/kong2024_dpadmm.svg)

## Results

| Rank $r$ of $A_3$ | Solver | Status | Outer Iters | Oracle Evals | Opt Residual | Feas Residual | Wall (s) |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `DPADMM` | CONVERGED | 11275 | 12498187 | 0.00e+00 | 1.91e-08 | 2.271 |
| 1 | `PAdaptADMM` | MAX_ITER_REACHED | 16000 | 171738504 | 3.64e+17 | 8.91e-07 | 5.389 |
| 1 | `ADMM (rho=5)` | CONVERGED | 245 | 3452 | 0.00e+00 | 3.14e-04 | 0.219 |
| 2 | `DPADMM` | CONVERGED | 13322 | 22497378 | 0.00e+00 | 8.82e-08 | 2.483 |
| 2 | `PAdaptADMM` | MAX_ITER_REACHED | 16000 | 99831727 | 9.09e+16 | 2.38e-07 | 3.635 |
| 2 | `ADMM (rho=5)` | CONVERGED | 415 | 5776 | 0.00e+00 | 3.93e-04 | 0.195 |
| 4 | `DPADMM` | CONVERGED | 12876 | 15917610 | 0.00e+00 | 1.44e-07 | 2.246 |
| 4 | `PAdaptADMM` | MAX_ITER_REACHED | 16000 | 177482353 | 2.40e+17 | 4.62e-07 | 5.371 |
| 4 | `ADMM (rho=5)` | CONVERGED | 280 | 3952 | 0.00e+00 | 7.17e-04 | 0.198 |
| 6 (full) | `DPADMM` | CONVERGED | 11192 | 12164808 | 0.00e+00 | 7.39e-08 | 2.120 |
| 6 (full) | `PAdaptADMM` | MAX_ITER_REACHED | 16000 | 263965258 | 1.68e+17 | 6.06e-07 | 7.461 |
| 6 (full) | `ADMM (rho=5)` | MAX_ITER_REACHED | 1500 | 20518 | 1.36e+00 | 1.94e-03 | 0.196 |

## Observations

- `DPADMM` converges across all four coupling ranks ($r \in \{1, 2, 4, 6\}$) to machine-zero stationarity and $O(10^{-8})$–$O(10^{-7})$ feasibility in ~2.1–2.5s on CPU, matching its rank-free $O(\varepsilon^{-3})$ guarantee.
- Fixed-penalty `ADMM (rho=5)` is fast when it converges ($r \in \{1, 2, 4\}$) but stalls at $r=6$ (`opt_residual = 1.36`), while `PAdaptADMM` (`gamma_init=10.0, lmt_bound=10.0`) diverges in primal stationarity across all four ranks on this coupled indefinite QP.
