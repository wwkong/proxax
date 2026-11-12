"""Certified proximal-point solvers for composite optimization."""

from __future__ import annotations

from proxax.solvers import base
from proxax.solvers import constrained
from proxax.solvers import inner
from proxax.solvers import unconstrained

ACGConfig = inner.ACGConfig
ACGState = inner.ACGState
ACGStatus = inner.ACGStatus
AG = unconstrained.AG
AGState = unconstrained.AGState
AIPP = unconstrained.AIPP
AIPPState = unconstrained.AIPPState
APD = unconstrained.APD
APDState = unconstrained.APDState
ECG = unconstrained.ECG
ECGState = unconstrained.ECGState
IALM = constrained.IALM
IALMState = constrained.IALMState
NCFISTA = unconstrained.NCFISTA
NCFISTAState = unconstrained.NCFISTAState
PFACGConfig = inner.PFACGConfig
PFACGState = inner.PFACGState
PFACGStatus = inner.PFACGStatus
QPAIPP = constrained.QPAIPP
QPAIPPState = constrained.QPAIPPState
QPAPD = constrained.QPAPD
QPAPDState = constrained.QPAPDState
Solution = base.Solution
Solver = base.Solver
SolveStats = base.SolveStats
SolveStatus = base.SolveStatus
StoppingConfig = base.StoppingConfig
UPFAG = unconstrained.UPFAG
UPFAGState = unconstrained.UPFAGState

__all__ = [
    "ACGConfig",
    "ACGState",
    "ACGStatus",
    "AG",
    "AGState",
    "AIPP",
    "AIPPState",
    "APD",
    "APDState",
    "ECG",
    "ECGState",
    "IALM",
    "IALMState",
    "NCFISTA",
    "NCFISTAState",
    "PFACGConfig",
    "PFACGState",
    "PFACGStatus",
    "QPAIPP",
    "QPAIPPState",
    "QPAPD",
    "QPAPDState",
    "Solution",
    "SolveStats",
    "SolveStatus",
    "Solver",
    "StoppingConfig",
    "UPFAG",
    "UPFAGState",
    "constrained",
    "inner",
    "unconstrained",
]
