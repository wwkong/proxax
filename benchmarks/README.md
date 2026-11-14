# Proxax Benchmarks

Benchmarking suites for comparing Proxax solvers, `optax` baselines, and
external standard problem collections across `float32` and `float64`.

## Structure

- **`benchmarks/internal/`**: Self-contained Proxax-native suites (`suites.py`,
  `run_all.py`) with zero external dependencies beyond `jax` and `optax`.
- **`benchmarks/external/`**: External benchmark suites (e.g. `maros_meszaros.py`
  for the Maros–Meszaros QP collection) and `plot_*.py` SVG generators.
- **`benchmarks/papers/`**: End-to-end paper reproduction runners
  (`reproduce_mpc.py`).

## Running Internal Suites

Run all internal suites with default `float32`:

```bash
uv run python -m benchmarks.internal.run_all
```

Run a specific suite (`unconstrained`, `constrained`, `spectral`, `vmap`) or
enable `float64`:

```bash
JAX_ENABLE_X64=1 uv run python -m benchmarks.internal.run_all --suite unconstrained --opt-tol 1e-4
```

Export structured JSON metrics:

```bash
uv run python -m benchmarks.internal.run_all --json-out benchmarks_out.json
```

## Running External Suites (Maros–Meszaros QPs)

Run a small curated subset of Maros–Meszaros QPs (downloads `.SIF` files into
`~/.cache/proxax/maros_meszaros/` on first run, or pass `--data-dir`):

```bash
JAX_ENABLE_X64=1 uv run python -m benchmarks.external.maros_meszaros --max-dim 200 --limit 10
```

## Reproducing Paper Benchmarks (`benchmarks/papers/`)

`benchmarks/papers/reproduce_mpc.py` runs the CPU and GPU external benchmark
suites and regenerates all Mocha-themed SVG plots in
`docs-experiments/_static/plots/`:

```bash
# Preview all commands without executing
uv run python -m benchmarks.papers.reproduce_mpc --dry-run

# Run CPU suites and regenerate plots
uv run python -m benchmarks.papers.reproduce_mpc --suite cpu

# Run GPU suites and regenerate GPU plots (requires CUDA-enabled jaxlib)
uv run python -m benchmarks.papers.reproduce_mpc --suite gpu

# Run everything (CPU + GPU)
uv run python -m benchmarks.papers.reproduce_mpc --suite all
```

Optional flags: `--json-dir` (default `/tmp/proxax_paper_results`),
`--plots-dir` (default `docs-experiments/_static/plots`), and `--skip-plots`.
