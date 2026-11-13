# Experiments

This directory is for paper replications, large-scale parameter sweeps, and
long-running numerical studies built on `proxax`.

Unlike `benchmarks/` (which holds fast, self-contained regression and
comparison suites), `experiments/` is meant for:

- Multi-seed and large-dimension scaling runs
- Exact paper table/figure replications
- Scripts that emit raw logs, CSV/JSON/NPZ artifacts, or plots

## Suggested Layout

Create one subdirectory per study or paper, for example:

```text
experiments/
  kong2021_dpadmm/
    run.py
    plot.py
    results/   # gitignored
  maia2026_p_adapt_admm/
    run.py
    results/   # gitignored
```

`results/`, `outputs/`, and `data/` subdirectories under `experiments/` are
gitignored by default so raw artifacts stay out of version control.

## Running

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python experiments/<study>/run.py
```
