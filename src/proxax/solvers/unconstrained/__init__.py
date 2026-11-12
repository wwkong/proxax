"""Unconstrained composite proximal point solvers."""

from __future__ import annotations

from proxax.solvers.unconstrained import aipp

AIPP = aipp.AIPP
AIPPState = aipp.AIPPState

__all__ = ["AIPP", "AIPPState"]
