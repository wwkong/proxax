# Changelog

All notable changes to **Proxax** are documented in this file.

## [Unreleased]

## [0.3.0] - 2026-11-13

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
