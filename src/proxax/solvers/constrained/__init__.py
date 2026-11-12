"""Constrained nonconvex composite optimization solvers."""

from __future__ import annotations

from proxax.solvers.constrained import iaipal
from proxax.solvers.constrained import ialm
from proxax.solvers.constrained import qp_aipp
from proxax.solvers.constrained import qp_apd
from proxax.solvers.constrained import s_prox_alm

IAIPAL = iaipal.IAIPAL
IAIPALState = iaipal.IAIPALState
IALM = ialm.IALM
IALMState = ialm.IALMState
QPAIPP = qp_aipp.QPAIPP
QPAIPPState = qp_aipp.QPAIPPState
QPAPD = qp_apd.QPAPD
QPAPDState = qp_apd.QPAPDState
SProxALM = s_prox_alm.SProxALM
SProxALMState = s_prox_alm.SProxALMState

__all__ = [
    "IAIPAL",
    "IAIPALState",
    "IALM",
    "IALMState",
    "QPAIPP",
    "QPAIPPState",
    "QPAPD",
    "QPAPDState",
    "SProxALM",
    "SProxALMState",
]
