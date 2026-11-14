# Changelog

All notable changes to **Proxax** are documented in this file.

## [Unreleased]

## [0.3.7] - 2026-11-14

### Added
- Added built-in Modified Ruiz Equilibration (`px.ruiz_equilibrate`, `px.solve_preconditioned`, `px.PreconditionedProblem` in `proxax.precondition`), supporting both dense `jax.Array` and sparse `jax.experimental.sparse.BCOO` matrices without densification, plus User Guide and API documentation.
- Added `BCOO` support to `px.layers.QPLayer` (`Q` and `A` type annotations and tests).
- Added zero-densification sparse QPSIF loader (`load_qpsif_sparse`, `ruiz_equilibrate_sparse`), vectorized sparse `Clarabel` setup, and large-scale Maros–Mészáros benchmark (`benchmarks/external/maros_meszaros_large.py`).

### Fixed
- Fixed `px.batch_implicit_solve(..., backend="threads")` with batched `BCOO` (`n_batch >= 1`) parameters by unwrapping `BCOO` cotangents before `jax.vmap` unbatching and reconstructing `float0`-indexed `BCOO` cotangents in `_bwd`.

## [0.3.6] - 2026-11-14

### Added
- Added native `jax.experimental.sparse.BCOO` support in `MatrixLinearOperator` and `px.constraints.as_linear_operator`, enabling zero-densification sparse `BCOO @ x` and `BCOO.T @ y` matvecs across all `px.constraints` builders (`equality`, `less_equal`, `greater_equal`, `box`, `linear_cone`) and constrained solvers.
- Added `--sparse` (`BCOO`) support to `benchmarks/external/maros_meszaros.py` and `benchmarks/external/maros_meszaros_medium.py`.
- Added large-scale sparse `BCOO` vs. dense vs. `Clarabel` external benchmark (`benchmarks/external/sparse_bcoo_vs_clarabel.py`) and report (`docs-experiments/sparse_bcoo_vs_clarabel.md`), showing up to **21.70×** speedup over dense `proxax-IALM` and **8.69×** speedup over `Clarabel` at $n = 4000$.
- Documented `BCOO` sparse matrices and constraints in `docs/guide/prox_and_constraints.md` and `docs/examples/constrained.md`.

### Fixed
- Treated `jax.experimental.sparse.BCOO` as an atomic leaf in `_slice_pytree` and `_slice_range_pytree` (`src/proxax/batch.py`) so `px.batch_solve(..., backend="threads")` and `chunk_size > 1` work with batched `BCOO` (`n_batch >= 1`) parameter PyTrees.

## [0.3.5] - 2026-11-14

### Added
- Added `px.batch_solve` and `px.batch_implicit_solve` (`proxax.batch`, also exported at top-level `proxax`) with `backend="auto" | "vmap" | "threads"`, persistent `ThreadPoolExecutor` (`_THREAD_POOL_CACHE`), `chunk_size` partitioning, and weak-keyed compiled caches (`_BATCH_SOLVE_CACHE`, `_BATCH_IMPLICIT_CACHE`).
- Added external benchmark suites and reports in `docs-experiments/` comparing `proxax` against `Clarabel`, `alpaqa`, and `cvxpylayers` (`batch_vs_clarabel`, `batch_vs_alpaqa`, `vmap_implicit_vs_cvxpylayers`, `mpc_rollout`, `diff_training_loop`, `large_matrix_free`).
- Separated **External Benchmarks** and **Internal Benchmarks** in `docs-experiments/index.md` with Executive Summary tables across all four sections.
- Separated `docs/guide/batching_and_parallelism.md` (`Batching & Parallelism`) from `docs/guide/optax_and_jax.md`, added `proxax.batch` to `docs/api.md`, and added `docs/appendices/implementation_optimizations.md` (`Appendix D: Implementation Optimizations`).

### Fixed
- Pre-compiled `jax.jit(_one_fwd).lower(...).compile()` outside `jax.pure_callback` and pre-converted `x_init` / `params_batch` to NumPy arrays before `pool.map` in `_host_batch_fwd`, eliminating nested JIT and `lax.slice` contention across worker threads.
- Added tracer-aware dispatch (`_run_fwd_dispatch`) and pre-compiled `_vmap_bwd` in `px.batch_implicit_solve(..., backend="threads")` so eager `jax.value_and_grad` loops bypass `jax.pure_callback` without requiring `block_until_ready()`.

## [0.3.1] - 2026-11-13

### Changed
- Reordered and polished the main documentation landing page (`docs/index.md`, `README.md`), enabled `home_page_in_toc`, added dark-mode `pygments_dark_style = "monokai"`, table hover styling, and SVG borders.
- Expanded the `docs-experiments/` replication suite with sequential-CPU replications (`kong2021_minmax`, `kong2022_spectral`, `kong2023_alm_vs_qp`, `kong2024_dpadmm`, `kong2024_spectrahedron`) and `DPADMM`-wins trajectory plots on `maia2026_table1_dqp`.

## [0.2.8] - 2026-11-13

### Changed
- Reorganized documentation for conciseness, clarity, and cleaner navigation: moved deep internal oracle/certificate math into `docs/appendices/oracles_and_certificates.md` and tightened `docs/foundations/*`, `docs/guide/*`, and `docs/quickstart.md`.
- Folded the 5 small `docs/foundations/problem_classes/*` subpages directly into `docs/foundations/problem_classes.md`.
- Grouped `docs/examples/index.md`, `docs/api.md`, and the external `Experiment Reports & Benchmarks` link under a unified `Reference & Resources` sidebar section in `docs/index.md`.

## [0.2.7] - 2026-11-13

### Added
- Added `proxax.layers` (`px.layers`: `ProxLayer`, `CappedSimplexLayer`, `SimplexLayer`, `BoxLayer`, `QPLayer`) providing drop-in differentiable optimization layers with `.batched()` (`jax.vmap`) support, tested in `tests/test_layers.py`.
- Added conic optimization benchmark suite (`experiments/conic_benchmarks/run.py`, `docs-experiments/conic_benchmarks.md`, `docs-experiments/_static/plots/conic_benchmark_profile.svg`) evaluating `PANOC-ALM` across `ExpCone`, `SOC`, and `Spectrahedron` problems.
- Added a Solver Selection Flowchart in `docs/guide/solving_problems.md` and `px.layers` documentation in `docs/api.md` and `docs/guide/optax_and_jax.md`.

## [0.2.6] - 2026-11-13

### Changed
- Split experiment benchmarks and scaling studies into a standalone Sphinx subproject under `docs-experiments/` with `docs-experiments/.readthedocs.yaml` (`https://proxax.readthedocs.io/projects/experiments/en/latest/`).
- Restored `sphinx_book_theme` with `show_navbar_depth=1` (collapsed left sidebar by default) on the main docs, and cross-linked the Experiments portal in `docs/index.md`.
- Re-rendered `docs/_static/plots/minmax_phase.gif` at 240 DPI (`1320x1152`) and kept raster GIF/PNG plots at native crisp resolution in `custom.css`.

## [0.2.5] - 2026-11-13

### Added
- Added 3D primal exponential cone projection (`px.prox.ExpCone` and `px.constraints.ExponentialCone`) with Moreau decomposition verification in `tests/test_prox_extra.py`.
- Added end-to-end JAX/Optax neural network structured layer example `examples/nn_structured_layer.py` using `jax.vmap` + `px.implicit_solve(..., method="panoc")`.
- Added interactive client-side column sorting for all Sphinx documentation tables (`docs/_static/sortable_tables.js` and `docs/_static/custom.css`).
- Updated `docs/guide/solving_problems.md`, `docs/guide/optax_and_jax.md`, `docs/guide/prox_and_constraints.md`, and `docs/api.md` with `ExpCone`, `ExponentialCone`, `panoc_alm`, and `proxax.prox.exotic` API docs.

## [0.2.4] - 2026-11-13

### Added
- Added `Hyperplane`, `Halfspace`, `Ellipsoid` (axis-aligned exact bisection projection), and `Huber` proximal operators in `proxax.prox`, with unit tests in `tests/test_prox_extra.py`.
- Added constrained `px.implicit_solve(..., method="panoc_alm")` verification test in `tests/test_prox_extra.py`.
- Added `PANOC-ALM` constrained benchmark suite (`experiments/panoc_alm_benchmarks/run.py`, `docs/experiments/panoc_alm_benchmarks.md`, `docs/_static/plots/panoc_alm_profile.svg`).
- Added batched `jax.vmap` + `px.implicit_solve` throughput benchmark (`experiments/vmap_implicit_benchmarks/run.py`, `docs/experiments/vmap_implicit_benchmarks.md`, `docs/_static/plots/vmap_implicit_scaling.svg`).

## [0.2.3] - 2026-11-13

### Added
- Wired `PANOC` as an inner subproblem solver in `IALM` (`subproblem_solver="panoc"` and `px.solve(..., method="panoc_alm")`), with optional `lipschitz_init` and `opt_tol_override` overrides on `PANOC.solve_oracle`.
- Added ill-conditioned & exotic-prox benchmark suite in `experiments/panoc_benchmarks/run.py`, report in `docs/experiments/panoc_benchmarks.md`, and performance profile plot in `docs/_static/plots/panoc_benchmark_profile.svg`.
- Added differentiable optimization layer example `examples/diff_capped_simplex.py` combining `px.implicit_solve(..., method="panoc")`, `CappedSimplex`, and `optax.adam`.

## [0.2.2] - 2026-11-13

### Added
- Added `px.solvers.PANOC` (`method="panoc"`), a JIT- and `vmap`-compatible composite quasi-Newton solver combining proximal gradient steps with L-BFGS directions over the Forward-Backward Envelope.
- Added `examples/exotic_and_panoc.py` and `docs/_static/plots/panoc_lasso.svg` showcasing `PANOC`, `Spectrahedron`, and `CappedSimplex`.
- Added unit tests in `tests/test_panoc.py`.

## [0.2.1] - 2026-11-13

### Added
- Added 11 JIT-compatible proximal operators in `proxax.prox.exotic` (re-exported under `px.prox`): `PSDCone`, `Spectrahedron`, `SOC`, `Isotonic`, `OWL`, `OSCAR`, `CappedSimplex`, `LogBarrier`, `NegEntropy`, `L0`, and `CardinalityBall`.
- Added unit tests in `tests/test_exotic_prox.py` and documentation in `docs/guide/prox_and_constraints.md`.

## [0.2.0] - 2026-11-13

### Added
- Extended `px.implicit_solve` (`proxax.implicit`) to support `MinMaxProblem` and `BlockProblem` (including `return_dual=True` for multi-block multipliers) via a shared `_implicit_vjp_solve` adjoint helper.
- Added `examples/bilevel_optax.py` and `docs/_static/plots/bilevel_optax.svg` demonstrating bilevel Elastic-Net hyperparameter tuning with `px.implicit_solve` and `optax.adam`.
- Added unit tests for `MinMaxProblem` (`ExtraGradient`) and `BlockProblem` (`DPADMM`) implicit differentiation in `tests/test_implicit.py`.

## [0.1.4] - 2026-11-13

### Added
- `px.plotting.plot_eval_breakdown`, `px.plotting.plot_tolerance_scaling`, `px.plotting.plot_phase_trajectory`, `px.plotting.animate_phase_trajectory`, and `px.plotting.plot_sparse_recovery` with Catppuccin `"mocha"` styling.
- `record_iterates=True` option on `px.HistoryTracker` for capturing iterate trajectories.
- Documentation sections 5–8 in `docs/guide/plotting.md` with SVGs and animated min-max phase portrait GIF.

## [0.1.3] - 2026-11-13

### Added
- Concise, tracer-safe `__repr__` implementations for `px.Solution`, `px.SolveStats`, and `px.solvers.OracleCounts`.
- `px.HistoryTracker` callback for recording per-iteration objective values, residuals, tolerances, oracle counts, and wall-clock times (with optional `verbose=True` table output).
- Tutorial section in `docs/guide/solving_problems.md` and runnable example `examples/history_tracker.py`.

### Changed
- Refactored core modules from code audit: split `optax_compat/transforms.py` into focused submodules, extracted shared `_penalty_common.py` and `_icg_common.py` helpers, replaced `if/elif` chains in `solve.py` with dispatch tables, decomposed `oracle/spectral.py`, trimmed `solvers/base.py` via `OracleCountsMixin`, and extracted `PAdaptADMM` module-level helpers.

## [0.1.2] - 2026-11-13

### Added
- `examples/minmax.py` and `examples/block.py` runnable examples for `px.MinMaxProblem` and `px.BlockProblem`.
- Sphinx `Experiments` documentation section with reproducible DQP benchmark report (`maia2026_table1_dqp`).
- Project logo (`docs/_static/logo.png`) and Read the Docs theme/sidebar enhancements.

### Fixed
- `PAdaptADMM` parameter handling and `adaptive_tau` bisection behavior.

### Changed
- Renamed `examples/unconstrained_nc_opt.py` and `examples/constrained_nc_opt.py` to `examples/unconstrained.py` and `examples/constrained.py`.
- Replaced `ROADMAP.md` with `CHANGELOG.md`.

## [0.1.1] - 2026-11-12

### Added
- Multi-block proximal ADMM solvers `DPADMM` and `PAdaptADMM` (with adaptive $\tau$ bisection) for `px.BlockProblem`.
- Multi-block DQP benchmark suite and experiment scripts (`experiments/maia2026_table1_dqp`).

## [0.1.0] - 2026-11-12

### Added
- `px.BlockProblem`, `px.BlockTerm`, and certified `ADMM` solver for multi-block linearly-constrained optimization.
- Additional proximal operators and sets: folded-concave decompositions, `PSDSet`, `Fantope`, `L21`, `LInf`, `LInfBall`, and `TV1D`.
- CI smoke checks executing all scripts in `examples/`.

## [0.0.5] - 2026-11-12

### Added
- `px.implicit_solve` for differentiating through solver solutions.
- Expanded `MinMaxProblem` and `MinMaxOracle` documentation, guides, and API reference.

## [0.0.4] - 2026-11-12

### Added
- `px.MinMaxProblem`, `MinMaxOracle`, primal-dual stationarity certificates, and solvers `ExtraGradient`, `PDHG`, and `ProxPointMinMax`.
- Reproducible `benchmarks/` suite across problem classes and GitHub Actions CI workflow.

## [0.0.3] - 2026-11-12

### Fixed
- `float64` precision handling in `DAICG` and composable state trackers.
- Documentation LaTeX formatting and example extraction (`px.constraints.identity`).

## [0.0.2] - 2026-11-12

### Changed
- Refactored inner, outer, and Optax solver states to composable `OracleCounts`, `SolverTracker`, and `ConstrainedTracker` structures.

## [0.0.1] - 2026-11-12

### Added
- Initial release of Proxax (`from proxax import px`) with `CompositeProblem` and `SpectralProblem`, 16 first-order proximal, augmented Lagrangian, and spectral solvers, PyTree-native oracles, and `px.optax_compat`.
