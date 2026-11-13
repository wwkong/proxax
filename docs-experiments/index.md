# Experiment Reports

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

## Scaling Studies

Accelerator (`vmap` / `jit` / GPU) throughput and batch-scaling studies:

* [`PAdaptADMM` CPU/GPU & `jax.vmap` Scaling Study](padapt_vmap_gpu.md)
* [Batched Differentiable Layers (`jax.vmap` + `implicit_solve`)](vmap_implicit_benchmarks.md)

## Literature Replications

Reproductions of published benchmark tables using `proxax`:

* [`PAdaptADMM` CPU Benchmarks (Maia et al., 2026 — Table 1 DQP)](maia2026_table1_dqp.md)

```{toctree}
:maxdepth: 1
:hidden:
:caption: Benchmarks

panoc_benchmarks
panoc_alm_benchmarks
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
```
