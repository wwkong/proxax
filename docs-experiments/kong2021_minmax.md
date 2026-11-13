# Nonconvex-Concave Min-Max Optimization (Kong & Monteiro, SIOPT 2021)

Small-scale sequential-CPU replication of **Kong & Monteiro (SIOPT 2021)**, *An accelerated inexact proximal point method for solving nonconvex-concave min-max problems*, comparing `ProxPointMinMax` against direct primal-dual methods (`ExtraGradient` and `PDHG`).

## Setup & Reproduction

All runs execute sequentially on CPU via [`experiments/kong2021_minmax/run.py`](https://github.com/wwkong/proxax/blob/main/experiments/kong2021_minmax/run.py) (`opt_tol = 1e-4`):

```bash
JAX_PLATFORMS=cpu uv run python experiments/kong2021_minmax/run.py
```

- **Nonconvex-Concave Min-Max QP (`16x16`):** $\min_{x \in [-1, 1]^{16}} \max_{y \in \Delta_{16}} \frac{1}{2}x^\top Q_x x + x^\top K y - \frac{1}{2}y^\top Q_y y + c_x^\top x - c_y^\top y$ with indefinite $Q_x$ (negative eigenvalues in $[-2.0, -0.5]$) and $Q_y \succ 0$.

## Results

| Problem | Solver | Status | Outer Iters | Oracle Evals | Objective | Opt Residual | Wall (s) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Nonconvex-Concave Min-Max QP (`16x16`) | `ProxPointMinMax` | CONVERGED | 15 | 5258 | -8.988911 | 9.56e-05 | 0.910 |
| Nonconvex-Concave Min-Max QP (`16x16`) | `ExtraGradient` | CONVERGED | 77 | 662 | -8.388439 | 9.07e-05 | 0.124 |
| Nonconvex-Concave Min-Max QP (`16x16`) | `PDHG` | CONVERGED | 60 | 462 | -8.388439 | 5.38e-05 | 0.166 |

## Observations

- All three solvers converge cleanly in under 1 second on CPU.
- Direct primal-dual methods (`PDHG` and `ExtraGradient`) use ~8–11x fewer oracle evaluations (462 and 662 vs. 5,258), while `ProxPointMinMax` settles at a **lower stationary objective** (`-8.989` vs. `-8.388`) on this nonconvex-concave landscape.
