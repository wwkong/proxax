"""Certified first-order proximal point methods for composite optimization."""

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

__version__ = "0.1.0"

AG = solvers.AG
AIPP = solvers.AIPP
APD = solvers.APD
CompositeProblem = _problem.CompositeProblem
ECG = solvers.ECG
IALM = solvers.IALM
KKTResiduals = _problem.KKTResiduals
QPAIPP = solvers.QPAIPP
QPAPD = solvers.QPAPD
Solution = solvers.Solution
SolveStats = solvers.SolveStats
SolveStatus = solvers.SolveStatus
Solver = solvers.Solver
StoppingConfig = solvers.StoppingConfig
solve = _solve.solve

__all__ = [
    "AG",
    "AIPP",
    "APD",
    "CompositeProblem",
    "ECG",
    "IALM",
    "KKTResiduals",
    "QPAIPP",
    "QPAPD",
    "Solution",
    "SolveStats",
    "SolveStatus",
    "Solver",
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
]
