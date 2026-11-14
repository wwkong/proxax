# Proxax Benchmarks

Benchmarking suites for comparing Proxax solvers, `optax` baselines, and
external standard problem collections across `float32` and `float64`.

## Structure

- **`benchmarks/internal/`**: Self-contained Proxax-native suites (`suites.py`,
  `run_all.py`) with zero external dependencies beyond `jax` and `optax`.
- **`benchmarks/external/`**: External benchmark suites (e.g. `maros_meszaros.py`
  for the Maros–Meszaros QP collection).

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
