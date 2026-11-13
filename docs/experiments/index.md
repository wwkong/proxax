# Experiment Reports

This section collects writeups, tables, and replication notes for large-scale
and paper-replication studies run under `experiments/` in the repository.

Each report documents:

- **Problem setup & citations** (exact formulation, dimensions, tolerances,
  and seeds)
- **Reproduction command** (`experiments/<study>/run.py`)
- **Summary tables & observations** (iteration counts, inner ACG/FISTA calls,
  wall-clock times, and stationarity/feasibility residuals)

* [`PAdaptADMM` CPU Benchmarks](maia2026_table1_dqp.md)
* [`PAdaptADMM` CPU/GPU & `jax.vmap` Scaling Study](padapt_vmap_gpu.md)

```{toctree}
:maxdepth: 1
:hidden:

maia2026_table1_dqp
padapt_vmap_gpu
```
