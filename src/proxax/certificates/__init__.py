"""Stationarity and KKT certificate refinement routines."""

from __future__ import annotations

from proxax.certificates import stationarity

RefinementResult = stationarity.RefinementResult
refine = stationarity.refine

__all__ = [
    "RefinementResult",
    "refine",
    "stationarity",
]
