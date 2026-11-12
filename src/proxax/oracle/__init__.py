"""Composite oracle abstractions and parameterized transformations."""

from __future__ import annotations

from proxax.oracle import composite
from proxax.oracle import terms

CompositeOracle = composite.CompositeOracle
CompositeOracleProtocol = composite.CompositeOracleProtocol
PenalizedOracle = composite.PenalizedOracle
ProxRegularizedOracle = composite.ProxRegularizedOracle
RescaledProxOracle = composite.RescaledProxOracle
SmoothObjective = terms.SmoothObjective
SmoothTerm = terms.SmoothTerm
ZeroSmooth = terms.ZeroSmooth
as_smooth_term = terms.as_smooth_term

__all__ = [
    "CompositeOracle",
    "CompositeOracleProtocol",
    "PenalizedOracle",
    "ProxRegularizedOracle",
    "RescaledProxOracle",
    "SmoothObjective",
    "SmoothTerm",
    "ZeroSmooth",
    "as_smooth_term",
]
