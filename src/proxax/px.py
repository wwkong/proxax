"""Curated public API namespace (`from proxax import px`)."""

from __future__ import annotations

from proxax import certificates
from proxax import constraints
from proxax import optax_compat
from proxax import oracle
from proxax import problem as _problem
from proxax import prox
from proxax import solve as _solve
from proxax import solvers
from proxax import tree
from proxax import typing

__version__ = "0.0.3"

CompositeProblem = _problem.CompositeProblem
KKTResiduals = _problem.KKTResiduals
Solution = solvers.Solution
SolveStats = solvers.SolveStats
SolveStatus = solvers.SolveStatus
Solver = solvers.Solver
SpectralOracle = oracle.SpectralOracle
SpectralProblem = oracle.SpectralProblem
StoppingConfig = solvers.StoppingConfig
solve = _solve.solve

__all__ = [
    "CompositeProblem",
    "KKTResiduals",
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
    "optax_compat",
    "oracle",
    "prox",
    "solve",
    "solvers",
    "tree",
    "typing",
]
