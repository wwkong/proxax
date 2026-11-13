# Multi-Block Nonconvex Splitting: `DPADMM` vs. `PAdaptADMM` (Kong & Monteiro, SIOPT 2024)

Small-scale sequential-CPU replication of **Kong & Monteiro (SIOPT 2024)**, *Global complexity bound of a proximal ADMM for linearly-constrained nonseperable nonconvex composite programming* (`DPADMM`), compared against **Maia et al. (2026)** (`PAdaptADMM`) and fixed-penalty `ADMM`.

## Setup & Reproduction

All runs execute sequentially on CPU via [`experiments/kong2024_dpadmm/run.py`](https://github.com/wwkong/proxax/blob/main/experiments/kong2024_dpadmm/run.py) (`opt_tol = 1e-3`, `feas_tol = 1e-3`):

```bash
JAX_PLATFORMS=cpu uv run python experiments/kong2024_dpadmm/run.py
```

Both problems split a 3-block nonconvex quadratic over $[-1, 1]^8$ per block ($n=24$ total) coupled by 6 linear equality constraints:
1. **Full-rank coupling:** All three block constraint matrices $A_1, A_2, A_3 \in \mathbb{R}^{6 \times 8}$ have full row rank.
2. **Rank-deficient coupling:** $A_3$ has rank $2 < 6$, testing convergence without rank assumptions.

## Results

| Problem | Solver | Status | Outer Iters | Oracle Evals | Opt Residual | Feas Residual | Wall (s) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 3-Block Nonconvex QP (full rank, `n=24`) | `DPADMM` | CONVERGED | 11192 | 12164808 | 0.00e+00 | 7.39e-08 | 2.276 |
| 3-Block Nonconvex QP (full rank, `n=24`) | `PAdaptADMM` | MAX_ITER_REACHED | 16000 | 365588160 | 9.58e+16 | 4.58e-07 | 9.932 |
| 3-Block Nonconvex QP (full rank, `n=24`) | `ADMM (rho=5)` | MAX_ITER_REACHED | 1500 | 20518 | 1.36e+00 | 1.94e-03 | 0.219 |
| 3-Block Nonconvex QP (rank-deficient, `n=24`) | `DPADMM` | CONVERGED | 13322 | 22497378 | 0.00e+00 | 8.82e-08 | 2.489 |
| 3-Block Nonconvex QP (rank-deficient, `n=24`) | `PAdaptADMM` | MAX_ITER_REACHED | 16000 | 227914521 | 2.00e+17 | 3.60e-07 | 6.661 |
| 3-Block Nonconvex QP (rank-deficient, `n=24`) | `ADMM (rho=5)` | CONVERGED | 415 | 5776 | 0.00e+00 | 3.93e-04 | 0.198 |

## Observations

- `DPADMM` converges reliably to machine-level stationarity and $O(10^{-8})$ feasibility on **both** the full-rank and rank-deficient instances in ~2.3–2.5s on CPU, matching its rank-free $O(\varepsilon^{-3})$ guarantee.
- Fixed-penalty `ADMM (rho=5)` stalls on the full-rank instance (`opt_residual = 1.36`), and `PAdaptADMM` with `lmt_bound=1.0` (below the $10.0$ upper curvature) diverges in primal stationarity, illustrating `DPADMM`'s robustness when curvature/rank assumptions are not met.
