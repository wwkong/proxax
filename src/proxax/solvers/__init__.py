"""Certified proximal-point solvers for composite optimization."""

from __future__ import annotations

from proxax.solvers import base
from proxax.solvers import constrained
from proxax.solvers import inner
from proxax.solvers import unconstrained

ACACG = unconstrained.ACACG
ACACGState = unconstrained.ACACGState
ACGConfig = inner.ACGConfig
ACGState = inner.ACGState
ACGStatus = inner.ACGStatus
AG = unconstrained.AG
AGState = unconstrained.AGState
AIPP = unconstrained.AIPP
AIPPState = unconstrained.AIPPState
APD = unconstrained.APD
APDState = unconstrained.APDState
AdapAPG = unconstrained.AdapAPG
AdapAPGState = unconstrained.AdapAPGState
AdapFISTA = unconstrained.AdapFISTA
AdapFISTAState = unconstrained.AdapFISTAState
ECG = unconstrained.ECG
ECGState = unconstrained.ECGState
IALM = constrained.IALM
IALMState = constrained.IALMState
IAICG = unconstrained.IAICG
IAICGState = unconstrained.IAICGState
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
    "ACACG",
    "ACACGState",
    "ACGConfig",
    "ACGState",
    "ACGStatus",
    "AG",
    "AGState",
    "AIPP",
    "AIPPState",
    "APD",
    "APDState",
    "AdapAPG",
    "AdapAPGState",
    "AdapFISTA",
    "AdapFISTAState",
    "ECG",
    "ECGState",
    "IALM",
    "IALMState",
    "IAICG",
    "IAICGState",
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
