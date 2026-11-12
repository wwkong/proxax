"""Certified proximal-point solvers for composite optimization."""

from __future__ import annotations

from proxax.solvers import base
from proxax.solvers import constrained
from proxax.solvers import inner
from proxax.solvers import unconstrained

ACGConfig = inner.ACGConfig
ACGState = inner.ACGState
ACGStatus = inner.ACGStatus
AIPP = unconstrained.AIPP
AIPPState = unconstrained.AIPPState
QPAIPP = constrained.QPAIPP
QPAIPPState = constrained.QPAIPPState
Solution = base.Solution
Solver = base.Solver
SolveStats = base.SolveStats
SolveStatus = base.SolveStatus
StoppingConfig = base.StoppingConfig

__all__ = [
    "ACGConfig",
    "ACGState",
    "ACGStatus",
    "AIPP",
    "AIPPState",
    "QPAIPP",
    "QPAIPPState",
    "Solution",
    "SolveStats",
    "SolveStatus",
    "Solver",
    "StoppingConfig",
    "constrained",
    "inner",
    "unconstrained",
]
