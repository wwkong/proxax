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

ACACG = solvers.ACACG
AG = solvers.AG
AIPP = solvers.AIPP
APD = solvers.APD
AdapAPG = solvers.AdapAPG
AdapFISTA = solvers.AdapFISTA
CompositeProblem = _problem.CompositeProblem
ECG = solvers.ECG
IALM = solvers.IALM
KKTResiduals = _problem.KKTResiduals
NCFISTA = solvers.NCFISTA
QPAIPP = solvers.QPAIPP
QPAPD = solvers.QPAPD
Solution = solvers.Solution
SolveStats = solvers.SolveStats
SolveStatus = solvers.SolveStatus
Solver = solvers.Solver
StoppingConfig = solvers.StoppingConfig
UPFAG = solvers.UPFAG
solve = _solve.solve

__all__ = [
    "ACACG",
    "AG",
    "AIPP",
    "APD",
    "AdapAPG",
    "AdapFISTA",
    "CompositeProblem",
    "ECG",
    "IALM",
    "KKTResiduals",
    "NCFISTA",
    "QPAIPP",
    "QPAPD",
    "Solution",
    "SolveStats",
    "SolveStatus",
    "Solver",
    "StoppingConfig",
    "UPFAG",
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
