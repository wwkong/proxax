# Changelog

All notable changes to **Proxax** are documented in this file.

## [Unreleased]

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
