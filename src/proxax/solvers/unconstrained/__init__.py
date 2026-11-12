"""Unconstrained composite proximal point solvers."""

from __future__ import annotations

from proxax.solvers.unconstrained import ac_acg
from proxax.solvers.unconstrained import adap_apg
from proxax.solvers.unconstrained import adap_fista
from proxax.solvers.unconstrained import ag
from proxax.solvers.unconstrained import aipp
from proxax.solvers.unconstrained import apd
from proxax.solvers.unconstrained import ecg
from proxax.solvers.unconstrained import ia_icg
from proxax.solvers.unconstrained import nc_fista
from proxax.solvers.unconstrained import upfag

ACACG = ac_acg.ACACG
ACACGState = ac_acg.ACACGState
AG = ag.AG
AGState = ag.AGState
AIPP = aipp.AIPP
AIPPState = aipp.AIPPState
APD = apd.APD
APDState = apd.APDState
AdapAPG = adap_apg.AdapAPG
AdapAPGState = adap_apg.AdapAPGState
AdapFISTA = adap_fista.AdapFISTA
AdapFISTAState = adap_fista.AdapFISTAState
ECG = ecg.ECG
ECGState = ecg.ECGState
IAICG = ia_icg.IAICG
IAICGState = ia_icg.IAICGState
NCFISTA = nc_fista.NCFISTA
NCFISTAState = nc_fista.NCFISTAState
UPFAG = upfag.UPFAG
UPFAGState = upfag.UPFAGState

__all__ = [
    "ACACG",
    "ACACGState",
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
    "IAICG",
    "IAICGState",
    "NCFISTA",
    "NCFISTAState",
    "UPFAG",
    "UPFAGState",
]
