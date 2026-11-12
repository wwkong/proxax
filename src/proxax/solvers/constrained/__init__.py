"""Constrained nonconvex composite optimization solvers."""

from __future__ import annotations

from proxax.solvers.constrained import qp_aipp

QPAIPP = qp_aipp.QPAIPP
QPAIPPState = qp_aipp.QPAIPPState

__all__ = [
    "QPAIPP",
    "QPAIPPState",
]
