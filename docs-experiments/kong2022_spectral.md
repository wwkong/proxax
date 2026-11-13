# Nonconvex Spectral Composite Optimization (Kong & Monteiro, COAP 2022)

Small-scale sequential-CPU replication of **Kong & Monteiro (COAP 2022)**, *Accelerated inexact composite gradient methods for nonconvex spectral optimization problems*, comparing doubly-accelerated (`DAICG`) and inner-accelerated (`IAICG`) methods.

## Setup & Reproduction

All runs execute sequentially on CPU via [`experiments/kong2022_spectral/run.py`](https://github.com/wwkong/proxax/blob/main/experiments/kong2022_spectral/run.py) (`opt_tol = 1e-3`, `max_outer_iters = 600`):

```bash
JAX_PLATFORMS=cpu uv run python experiments/kong2022_spectral/run.py
```

- **Spectral Matrix Completion (`20x20, rank=3`):** Noisy observed matrix recovery (`45%` sampling) with smooth Frobenius loss $f_1(X)$ and nonconvex log-sum singular-value regularization $f_2^{\mathcal{V}}(\sigma(X)) + h^{\mathcal{V}}(\sigma(X))$ on a Frobenius-norm ball.

## Results

| Problem | Solver | Status | Outer Iters | Oracle Evals | Objective | Opt Residual | Wall (s) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Spectral Matrix Completion (`20x20, rank=3`) | `DAICG (monotone)` | CONVERGED | 297 | 6917 | 2.171382 | 9.96e-04 | 1.898 |
| Spectral Matrix Completion (`20x20, rank=3`) | `DAICG (nonmonotone)` | CONVERGED | 322 | 7468 | 2.171396 | 9.77e-04 | 0.596 |
| Spectral Matrix Completion (`20x20, rank=3`) | `IAICG` | MAX_ITER_REACHED | 600 | 16166 | 2.529692 | 2.95e-02 | 0.532 |

## Observations

- Both `DAICG` variants converge to `opt_tol = 1e-3` in **297–322 outer iterations** (~6.9k–7.5k oracle evaluations), whereas `IAICG` hits the 600-iteration budget with `opt_residual = 2.95e-2` and a higher objective (`2.53` vs. `2.17`), confirming the practical advantage of outer (double) acceleration in `SpectralProblem`.
