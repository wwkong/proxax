"""Unconstrained composite proximal point solvers."""

from __future__ import annotations

from proxax.solvers.unconstrained import aipp
from proxax.solvers.unconstrained import apd
from proxax.solvers.unconstrained import ecg

AIPP = aipp.AIPP
AIPPState = aipp.AIPPState
APD = apd.APD
APDState = apd.APDState
ECG = ecg.ECG
ECGState = ecg.ECGState

__all__ = ["AIPP", "AIPPState", "APD", "APDState", "ECG", "ECGState"]
