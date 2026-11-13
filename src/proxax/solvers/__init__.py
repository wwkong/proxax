"""Certified proximal-point solvers for composite optimization."""

from __future__ import annotations

from proxax.solvers import base
from proxax.solvers import constrained
from proxax.solvers import inner
from proxax.solvers import minmax
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
ConstrainedTracker = base.ConstrainedTracker
DAICG = unconstrained.DAICG
DAICGState = unconstrained.DAICGState
ECG = unconstrained.ECG
ECGState = unconstrained.ECGState
ExtraGradient = minmax.ExtraGradient
ExtraGradientState = minmax.ExtraGradientState
IALM = constrained.IALM
IALMState = constrained.IALMState
IAICG = unconstrained.IAICG
IAICGState = unconstrained.IAICGState
IAIPAL = constrained.IAIPAL
IAIPALState = constrained.IAIPALState
NCFISTA = unconstrained.NCFISTA
NCFISTAState = unconstrained.NCFISTAState
OracleCounts = base.OracleCounts
PDHG = minmax.PDHG
PDHGState = minmax.PDHGState
PFACGConfig = inner.PFACGConfig
PFACGState = inner.PFACGState
PFACGStatus = inner.PFACGStatus
ProxPointMinMax = minmax.ProxPointMinMax
ProxPointMinMaxState = minmax.ProxPointMinMaxState
QPAIPP = constrained.QPAIPP
QPAIPPState = constrained.QPAIPPState
QPAPD = constrained.QPAPD
QPAPDState = constrained.QPAPDState
SProxALM = constrained.SProxALM
SProxALMState = constrained.SProxALMState
Solution = base.Solution
Solver = base.Solver
SolveStats = base.SolveStats
SolveStatus = base.SolveStatus
SolverTracker = base.SolverTracker
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
    "ConstrainedTracker",
    "DAICG",
    "DAICGState",
    "ECG",
    "ECGState",
    "ExtraGradient",
    "ExtraGradientState",
    "IALM",
    "IALMState",
    "IAICG",
    "IAICGState",
    "IAIPAL",
    "IAIPALState",
    "NCFISTA",
    "NCFISTAState",
    "OracleCounts",
    "PDHG",
    "PDHGState",
    "PFACGConfig",
    "PFACGState",
    "PFACGStatus",
    "ProxPointMinMax",
    "ProxPointMinMaxState",
    "QPAIPP",
    "QPAIPPState",
    "QPAPD",
    "QPAPDState",
    "SProxALM",
    "SProxALMState",
    "Solution",
    "SolveStats",
    "SolveStatus",
    "Solver",
    "SolverTracker",
    "StoppingConfig",
    "UPFAG",
    "UPFAGState",
    "constrained",
    "inner",
    "minmax",
    "unconstrained",
]
