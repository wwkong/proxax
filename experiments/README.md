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

## Experiment Reports (ReadTheDocs)

Write up results for each study as a Markdown report under
`docs/experiments/<study>.md` and add it to `docs/experiments/index.md` so it
appears in the **Experiments** section of the documentation.

## Running

```bash
uv run python experiments/<study>/run.py
```
