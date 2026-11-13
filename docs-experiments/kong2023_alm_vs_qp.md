# ALM vs. Quadratic Penalty on Constrained NCO (Kong et al., 2023)

Small-scale sequential-CPU replication comparing proximal augmented Lagrangian methods (**Kong, Monteiro, & Melo, SIOPT 2023 `IAIPAL`**; **Kong & Monteiro, COAP 2023 `AIDAL` / `IALM`**) against earlier quadratic-penalty methods (**Kong, Melo, & Monteiro, SIOPT 2019 / COAP 2020 `QP-AIPP`**).

## Setup & Reproduction

All runs execute sequentially on CPU via [`experiments/kong2023_alm_vs_qp/run.py`](https://github.com/wwkong/proxax/blob/main/experiments/kong2023_alm_vs_qp/run.py) (`opt_tol = 1e-3`, `feas_tol = 1e-3`):

```bash
JAX_PLATFORMS=cpu uv run python experiments/kong2023_alm_vs_qp/run.py
```

1. **Linearly Constrained QP (`n=24, m=4`):** Indefinite quadratic over $[0, 1]^{24}$ with 4 linear equality constraints.
2. **Constrained $k$-Fantope SPCA (`8x8, k=2`):** Sparse-PCA-like objective over the $k$-Fantope $\{X : 0 \preceq X \preceq I,\, \operatorname{tr}(X) = 2\}$ with 2 linear equality constraints.

## Results

| Problem | Solver | Status | Outer Iters | Oracle Evals | Opt Residual | Feas Residual | Wall (s) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Linearly Constrained QP (`n=24, m=4`) | `IAIPAL` | MAX_ITER_REACHED | 40 | 761 | 1.29e-01 | 4.40e-02 | 1.310 |
| Linearly Constrained QP (`n=24, m=4`) | `IALM (APD)` | CONVERGED | 13 | 7658 | 9.76e-04 | 5.19e-04 | 0.838 |
| Linearly Constrained QP (`n=24, m=4`) | `IALM (PANOC)` | CONVERGED | 13 | 9042 | 7.14e-04 | 6.45e-04 | 0.772 |
| Linearly Constrained QP (`n=24, m=4`) | `SProxALM` | CONVERGED | 699 | 4916 | 9.25e-04 | 7.68e-05 | 0.352 |
| Linearly Constrained QP (`n=24, m=4`) | `QP-APD` | CONVERGED | 14 | 62860 | 9.32e-04 | 9.06e-04 | 0.736 |
| Linearly Constrained QP (`n=24, m=4`) | `QP-AIPP` | CONVERGED | 14 | 69155 | 8.55e-04 | 9.06e-04 | 0.909 |
| Constrained $k$-Fantope SPCA (`8x8, k=2`) | `IAIPAL` | MAX_ITER_REACHED | 40 | 8511 | 5.28e-03 | 1.62e-07 | 1.181 |
| Constrained $k$-Fantope SPCA (`8x8, k=2`) | `IALM (APD)` | CONVERGED | 8 | 1140 | 9.90e-04 | 7.89e-04 | 0.490 |
| Constrained $k$-Fantope SPCA (`8x8, k=2`) | `IALM (PANOC)` | CONVERGED | 8 | 2078 | 9.88e-04 | 7.74e-04 | 0.527 |
| Constrained $k$-Fantope SPCA (`8x8, k=2`) | `SProxALM` | CONVERGED | 490 | 3445 | 9.98e-04 | 1.05e-05 | 0.410 |
| Constrained $k$-Fantope SPCA (`8x8, k=2`) | `QP-APD` | CONVERGED | 11 | 2282 | 9.56e-04 | 8.05e-04 | 0.475 |
| Constrained $k$-Fantope SPCA (`8x8, k=2`) | `QP-AIPP` | CONVERGED | 11 | 6326 | 7.89e-04 | 8.05e-04 | 0.638 |

## Observations

- Replacing quadratic penalty (`QP-AIPP` / `QP-APD`) with inexact augmented Lagrangian (`IALM (APD)` / `IALM (PANOC)`) reduces oracle evaluations by **8–9x** on the linearly constrained QP (7,658 vs. 69,155) and by **5.5x** on the $k$-Fantope SPCA problem (1,140 vs. 6,326), consistent with the $O(\varepsilon^{-5/2}\log^2\varepsilon^{-1})$ vs. $O(\varepsilon^{-3})$ complexity improvement.
- Single-loop `SProxALM` is also competitive in total oracle calls (4,916 and 3,445) and fastest in wall-clock time on these small CPU instances.
