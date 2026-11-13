"""Primal-dual saddle-point solvers for `MinMaxProblem`."""

from __future__ import annotations

from proxax.solvers.minmax import extra_gradient
from proxax.solvers.minmax import pdhg
from proxax.solvers.minmax import prox_point

ExtraGradient = extra_gradient.ExtraGradient
ExtraGradientState = extra_gradient.ExtraGradientState
PDHG = pdhg.PDHG
PDHGState = pdhg.PDHGState
ProxPointMinMax = prox_point.ProxPointMinMax
ProxPointMinMaxState = prox_point.ProxPointMinMaxState

__all__ = [
    "ExtraGradient",
    "ExtraGradientState",
    "PDHG",
    "PDHGState",
    "ProxPointMinMax",
    "ProxPointMinMaxState",
]
