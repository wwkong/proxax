# Experiment Reports

[← Back to Proxax Documentation](https://proxax.readthedocs.io/en/latest/)

This section collects writeups, tables, and replication notes for large-scale
and paper-replication studies run under `experiments/` in the repository.

Each report documents:

- **Problem setup & citations** (exact formulation, dimensions, tolerances,
  and seeds)
- **Reproduction command** (`experiments/<study>/run.py`)
- **Summary tables & observations** (iteration counts, inner ACG/FISTA calls,
  wall-clock times, and stationarity/feasibility residuals)

## Benchmarks

Solver comparisons and Dolan-Moré performance profiles across ill-conditioned
and constrained composite suites:

* [`PANOC` Ill-Conditioned & Exotic-Prox Benchmarks](panoc_benchmarks.md)
* [`PANOC-ALM` Constrained Solver Benchmarks](panoc_alm_benchmarks.md)
* [Conic Optimization Benchmarks (`ExpCone`, `SOC`, `Spectrahedron`)](conic_benchmarks.md)
* [Maros–Meszaros QP Benchmarks](maros_meszaros.md)
* [Maros–Meszaros Medium-Scale QP Benchmarks](maros_meszaros_medium.md)
* [Matrix-Free CPU Benchmarks (`proxax` vs. `Clarabel`)](matrix_free_vs_clarabel.md)
* [Nonconvex CPU Benchmarks (`proxax` vs. `alpaqa`)](panoc_vs_alpaqa.md)
* [Batched Differentiable Layers (`proxax` vs. `cvxpylayers`)](vmap_implicit_vs_cvxpylayers.md)
* [Batched Convex Solves (`proxax` vs. `Clarabel`)](batch_vs_clarabel.md)
* [Batched Nonconvex Solves (`proxax` vs. `alpaqa`)](batch_vs_alpaqa.md)
* [Closed-Loop Warm-Started MPC Rollout (`proxax` vs. `Clarabel` & `alpaqa`)](mpc_rollout.md)
* [End-to-End Differentiable Training Loop (`proxax` vs. `cvxpylayers`)](diff_training_loop.md)

## Scaling Studies

Accelerator (`vmap` / `jit` / GPU) throughput and batch-scaling studies:

* [`PAdaptADMM` CPU/GPU & `jax.vmap` Scaling Study](padapt_vmap_gpu.md)
* [Batched Differentiable Layers (`jax.vmap` + `implicit_solve`)](vmap_implicit_benchmarks.md)

## Literature Replications

Reproductions of published benchmark tables using `proxax`:

* [`PAdaptADMM` CPU Benchmarks (Maia et al., 2026 — Table 1 DQP)](maia2026_table1_dqp.md)
* [Curvature-Free Nonconvex QPs (Kong, SIOPT 2024)](kong2024_spectrahedron.md)
* [ALM vs. Quadratic Penalty on Constrained NCO (Kong et al., 2023)](kong2023_alm_vs_qp.md)
* [Multi-Block Nonconvex Splitting: `DPADMM` vs. `PAdaptADMM` (Kong & Monteiro, SIOPT 2024)](kong2024_dpadmm.md)
* [Nonconvex Spectral Composite Optimization (Kong & Monteiro, COAP 2022)](kong2022_spectral.md)
* [Nonconvex-Concave Min-Max Optimization (Kong & Monteiro, SIOPT 2021)](kong2021_minmax.md)

```{toctree}
:maxdepth: 1
:hidden:
:caption: Benchmarks

panoc_benchmarks
panoc_alm_benchmarks
conic_benchmarks
maros_meszaros
maros_meszaros_medium
matrix_free_vs_clarabel
panoc_vs_alpaqa
vmap_implicit_vs_cvxpylayers
batch_vs_clarabel
batch_vs_alpaqa
mpc_rollout
diff_training_loop
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Scaling Studies

padapt_vmap_gpu
vmap_implicit_benchmarks
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Literature Replications

maia2026_table1_dqp
kong2024_spectrahedron
kong2023_alm_vs_qp
kong2024_dpadmm
kong2022_spectral
kong2021_minmax
```
