"""Unconstrained composite proximal point solvers."""

from __future__ import annotations

from proxax.solvers.unconstrained import ag
from proxax.solvers.unconstrained import aipp
from proxax.solvers.unconstrained import apd
from proxax.solvers.unconstrained import ecg
from proxax.solvers.unconstrained import upfag

AG = ag.AG
AGState = ag.AGState
AIPP = aipp.AIPP
AIPPState = aipp.AIPPState
APD = apd.APD
APDState = apd.APDState
ECG = ecg.ECG
ECGState = ecg.ECGState
UPFAG = upfag.UPFAG
UPFAGState = upfag.UPFAGState

__all__ = [
    "AG",
    "AGState",
    "AIPP",
    "AIPPState",
    "APD",
    "APDState",
    "ECG",
    "ECGState",
    "UPFAG",
    "UPFAGState",
]
