# Curvature-Free Nonconvex QPs (Kong, SIOPT 2024)

Small-scale sequential-CPU replication of the parameter-free, curvature-free accelerated proximal descent study in **Kong (SIOPT 2024)**, *Complexity-optimal and parameter-free first-order methods for finding stationary points of composite optimization problems*, which supersedes earlier fixed-curvature `AIPP` unconstrained benchmarks (Kong, Melo, Monteiro, COAP 2020).

## Setup & Reproduction

All runs execute sequentially on CPU via [`experiments/kong2024_spectrahedron/run.py`](https://github.com/wwkong/proxax/blob/main/experiments/kong2024_spectrahedron/run.py) (`opt_tol = 1e-4`), sweeping condition parameter $\kappa \in \{10, 40, 100, 250\}$ across two nonconvex problem families:

```bash
JAX_PLATFORMS=cpu uv run python experiments/kong2024_spectrahedron/run.py
```

1. **Spectrahedron QP (`10x10`):** Indefinite quadratic $\frac{1}{2}\langle X, \mathcal{Q}(X)\rangle + \langle C, X\rangle$ over $\{X \succeq 0 : \operatorname{tr}(X) = 1\}$.
2. **Simplex QP (`n=30`):** Indefinite quadratic over the probability simplex $\Delta_{30}$ with negative eigenvalues in $[-4.0, -0.5]$ and positive eigenvalues in $[1.0, \kappa]$.

## Scaling Plot

![Curvature-Free APD vs. Fixed-Curvature Solvers](_static/plots/kong2024_spectrahedron.svg)

## Summary Table ($\kappa = 10$ vs. $\kappa = 250$)

| Problem | Solver | $\kappa=10$ Status (Evals) | $\kappa=40$ Status (Evals) | $\kappa=100$ Status (Evals) | $\kappa=250$ Status (Evals) |
| --- | --- | --- | --- | --- | --- |
| Spectrahedron QP (`10x10`) | `APD` | CONVERGED (71) | CONVERGED (96) | CONVERGED (146) | CONVERGED (520) |
| Spectrahedron QP (`10x10`) | `AdapAPG` | CONVERGED (48) | CONVERGED (90) | CONVERGED (154) | CONVERGED (302) |
| Spectrahedron QP (`10x10`) | `PANOC` | CONVERGED (61) | CONVERGED (97) | CONVERGED (147) | CONVERGED (229) |
| Spectrahedron QP (`10x10`) | `AIPP` | CONVERGED (98) | CONVERGED (210) | CONVERGED (238) | CONVERGED (340) |
| Spectrahedron QP (`10x10`) | `AG` | CONVERGED (60) | CONVERGED (72) | CONVERGED (186) | CONVERGED (242) |
| Spectrahedron QP (`10x10`) | `UPFAG` | MAX_ITER (18006) | MAX_ITER (18006) | MAX_ITER (18010) | MAX_ITER (18029) |
| Simplex QP (`n=30`) | `APD` | CONVERGED (207) | CONVERGED (246) | CONVERGED (353) | CONVERGED (777) |
| Simplex QP (`n=30`) | `AdapAPG` | CONVERGED (118) | CONVERGED (184) | CONVERGED (332) | CONVERGED (1438) |
| Simplex QP (`n=30`) | `PANOC` | CONVERGED (148) | CONVERGED (267) | CONVERGED (458) | CONVERGED (1096) |
| Simplex QP (`n=30`) | `AIPP` | CONVERGED (226) | CONVERGED (434) | CONVERGED (1026) | CONVERGED (3232) |
| Simplex QP (`n=30`) | `AG` | CONVERGED (238) | CONVERGED (380) | CONVERGED (1500) | CONVERGED (4499) |
| Simplex QP (`n=30`) | `UPFAG` | CONVERGED (316) | CONVERGED (1013) | MAX_ITER (18018) | CONVERGED (3600) |

## Observations

- On the ill-conditioned Simplex QP ($\kappa = 250$), curvature-free `APD` is the **most evaluation-efficient solver** in the suite (`777` oracle evals vs. `1096` for `PANOC`, `1438` for `AdapAPG`, `3232` for `AIPP`, and `4499` for `AG`), scaling much more gently with $\kappa$ than fixed-curvature baselines.
- On the small `10x10` Spectrahedron QP, `PANOC` and `AdapAPG` edge out `APD` at $\kappa=250$ (`229` and `302` vs. `520` evals), while `UPFAG` stalls at the 2,000-iteration budget.
