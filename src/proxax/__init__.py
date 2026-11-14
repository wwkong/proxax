"""Certified first-order proximal point methods for composite optimization."""

from __future__ import annotations

from proxax import batch as _batch
from proxax import certificates
from proxax import constraints
from proxax import implicit
from proxax import layers
from proxax import optax_compat
from proxax import oracle
from proxax import plotting
from proxax import precondition
from proxax import problem as _problem
from proxax import prox
from proxax import px
from proxax import solve as _solve
from proxax import solvers
from proxax import tree
from proxax import typing

__version__ = px.__version__

BlockProblem = oracle.BlockProblem
BlockTerm = oracle.BlockTerm
CompositeProblem = _problem.CompositeProblem
HistoryTracker = solvers.HistoryTracker
KKTResiduals = _problem.KKTResiduals
MinMaxOracle = oracle.MinMaxOracle
MinMaxProblem = oracle.MinMaxProblem
PreconditionedProblem = precondition.PreconditionedProblem
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
ruiz_equilibrate = precondition.ruiz_equilibrate
solve = _solve.solve
solve_preconditioned = precondition.solve_preconditioned

__all__ = [
    "BlockProblem",
    "BlockTerm",
    "CompositeProblem",
    "HistoryTracker",
    "KKTResiduals",
    "MinMaxOracle",
    "MinMaxProblem",
    "PreconditionedProblem",
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
    "precondition",
    "prox",
    "px",
    "ruiz_equilibrate",
    "solve",
    "solve_preconditioned",
    "solvers",
    "tree",
    "typing",
]
