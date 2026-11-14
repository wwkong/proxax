"""Curated public API namespace (`from proxax import px`)."""

from __future__ import annotations

from proxax import batch as _batch
from proxax import certificates
from proxax import constraints
from proxax import implicit
from proxax import layers
from proxax import optax_compat
from proxax import oracle
from proxax import plotting
from proxax import problem as _problem
from proxax import prox
from proxax import solve as _solve
from proxax import solvers
from proxax import tree
from proxax import typing

__version__ = "0.3.6"

BlockProblem = oracle.BlockProblem
BlockTerm = oracle.BlockTerm
CompositeProblem = _problem.CompositeProblem
HistoryTracker = solvers.HistoryTracker
KKTResiduals = _problem.KKTResiduals
MinMaxOracle = oracle.MinMaxOracle
MinMaxProblem = oracle.MinMaxProblem
Solution = solvers.Solution
SolveStats = solvers.SolveStats
SolveStatus = solvers.SolveStatus
Solver = solvers.Solver
SpectralOracle = oracle.SpectralOracle
SpectralProblem = oracle.SpectralProblem
StoppingConfig = solvers.StoppingConfig
batch_implicit_solve = _batch.batch_implicit_solve
batch_solve = _batch.batch_solve
implicit_solve = implicit.implicit_solve
solve = _solve.solve

__all__ = [
    "BlockProblem",
    "BlockTerm",
    "CompositeProblem",
    "HistoryTracker",
    "KKTResiduals",
    "MinMaxOracle",
    "MinMaxProblem",
    "Solution",
    "SolveStats",
    "SolveStatus",
    "Solver",
    "SpectralOracle",
    "SpectralProblem",
    "StoppingConfig",
    "__version__",
    "batch_implicit_solve",
    "batch_solve",
    "certificates",
    "constraints",
    "implicit",
    "implicit_solve",
    "layers",
    "optax_compat",
    "oracle",
    "plotting",
    "prox",
    "solve",
    "solvers",
    "tree",
    "typing",
]
