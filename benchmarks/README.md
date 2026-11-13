# Proxax Benchmarks

Self-contained benchmarking suite for comparing Proxax solvers, `optax`
baselines, and batched `jax.jit(jax.vmap(...))` throughput across `float32`
and `float64`.

## Running

Run all suites with default `float32`:

```bash
uv run python -m benchmarks.run_all
```

Run a specific suite (`unconstrained`, `constrained`, `spectral`, `vmap`) or
enable `float64`:

```bash
JAX_ENABLE_X64=1 uv run python -m benchmarks.run_all --suite unconstrained --opt-tol 1e-4
```

Export structured JSON metrics:

```bash
uv run python -m benchmarks.run_all --json-out benchmarks_out.json
```
