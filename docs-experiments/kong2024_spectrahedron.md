# Curvature-Free Nonconvex QPs (Kong, SIOPT 2024)

Small-scale sequential-CPU replication of the parameter-free, curvature-free accelerated proximal descent study in **Kong (SIOPT 2024)**, *Complexity-optimal and parameter-free first-order methods for finding stationary points of composite optimization problems*, which supersedes earlier fixed-curvature `AIPP` unconstrained benchmarks (Kong, Melo, Monteiro, COAP 2020).

## Setup & Reproduction

All runs execute sequentially on CPU via [`experiments/kong2024_spectrahedron/run.py`](https://github.com/wwkong/proxax/blob/main/experiments/kong2024_spectrahedron/run.py) (`opt_tol = 1e-4`):

```bash
JAX_PLATFORMS=cpu uv run python experiments/kong2024_spectrahedron/run.py
```

1. **Spectrahedron QP (`10x10`):** Indefinite quadratic $\frac{1}{2}\langle X, \mathcal{Q}(X)\rangle + \langle C, X\rangle$ over $\{X \succeq 0 : \operatorname{tr}(X) = 1\}$.
2. **Simplex QP (`n=30, cond=40`):** Indefinite quadratic over the probability simplex $\Delta_{30}$ with negative eigenvalues in $[-4.0, -0.5]$ and positive eigenvalues in $[1.0, 40.0]$.

## Results

| Problem | Solver | Status | Outer Iters | Oracle Evals | Objective | Opt Residual | Wall (s) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Spectrahedron QP (`10x10`) | `APD` | CONVERGED | 5 | 91 | -4.398335 | 9.85e-05 | 1.367 |
| Spectrahedron QP (`10x10`) | `AdapAPG` | CONVERGED | 9 | 64 | -4.398338 | 8.63e-05 | 0.351 |
| Spectrahedron QP (`10x10`) | `AIPP` | CONVERGED | 4 | 134 | -4.398336 | 1.09e-05 | 0.348 |
| Spectrahedron QP (`10x10`) | `AG` | CONVERGED | 10 | 66 | -4.398335 | 6.39e-05 | 0.224 |
| Spectrahedron QP (`10x10`) | `UPFAG` | CONVERGED | 9 | 87 | -4.398340 | 8.17e-05 | 0.292 |
| Spectrahedron QP (`10x10`) | `PANOC` | CONVERGED | 6 | 72 | -4.398338 | 3.21e-05 | 0.382 |
| Simplex QP (`n=30, cond=40`) | `APD` | CONVERGED | 6 | 246 | -0.754351 | 9.66e-05 | 0.720 |
| Simplex QP (`n=30, cond=40`) | `AdapAPG` | CONVERGED | 21 | 184 | -0.754351 | 8.45e-05 | 0.254 |
| Simplex QP (`n=30, cond=40`) | `AIPP` | CONVERGED | 11 | 434 | -0.754351 | 6.75e-05 | 0.297 |
| Simplex QP (`n=30, cond=40`) | `AG` | CONVERGED | 61 | 380 | -0.754351 | 6.60e-05 | 0.193 |
| Simplex QP (`n=30, cond=40`) | `UPFAG` | CONVERGED | 111 | 1013 | -0.754351 | 1.14e-05 | 0.226 |
| Simplex QP (`n=30, cond=40`) | `PANOC` | CONVERGED | 23 | 267 | -0.754351 | 2.97e-05 | 0.290 |

## Observations

- On both small-scale nonconvex problems, curvature-free `APD` cuts total oracle evaluations by **~32–43%** compared to `AIPP` (91 vs. 134 on Spectrahedron QP; 246 vs. 434 on Simplex QP) without requiring prior curvature constants.
- `AdapAPG` and `PANOC` are also strong on these small instances (184 and 267 oracle evaluations on the conditioned simplex QP), while `UPFAG` and `AG` require substantially more evaluations as conditioning worsens.
