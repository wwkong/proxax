"""Stationarity and KKT certificate refinement routines."""

from __future__ import annotations

from proxax.certificates import minmax
from proxax.certificates import stationarity

MinMaxRefinementResult = minmax.MinMaxRefinementResult
RefinementResult = stationarity.RefinementResult
refine = stationarity.refine
refine_minmax = minmax.refine_minmax

__all__ = [
    "MinMaxRefinementResult",
    "RefinementResult",
    "minmax",
    "refine",
    "refine_minmax",
    "stationarity",
]
