"""Certified first-order proximal point methods for composite optimization."""

from __future__ import annotations

from proxax import certificates
from proxax import constraints
from proxax import implicit
from proxax import optax_compat
from proxax import oracle
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
implicit_solve = implicit.implicit_solve
solve = _solve.solve

__all__ = [
    "BlockProblem",
    "BlockTerm",
    "CompositeProblem",
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
    "certificates",
    "constraints",
    "implicit",
    "implicit_solve",
    "optax_compat",
    "oracle",
    "prox",
    "px",
    "solve",
    "solvers",
    "tree",
    "typing",
]
