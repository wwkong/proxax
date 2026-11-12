"""Proximal operators and PyTree combinators."""

from __future__ import annotations

from proxax.prox import base
from proxax.prox import combinators
from proxax.prox import indicators
from proxax.prox import norms

Box = indicators.Box
BoxL1 = indicators.BoxL1
ElasticNet = norms.ElasticNet
L1 = norms.L1
L1Ball = indicators.L1Ball
L2 = norms.L2
L2Ball = indicators.L2Ball
L2Squared = norms.L2Squared
MultiProx = combinators.MultiProx
NonNegative = indicators.NonNegative
NuclearNorm = norms.NuclearNorm
ProxOperator = base.ProxOperator
Shifted = base.Shifted
Simplex = indicators.Simplex
Zero = base.Zero

__all__ = [
    "Box",
    "BoxL1",
    "ElasticNet",
    "L1",
    "L1Ball",
    "L2",
    "L2Ball",
    "L2Squared",
    "MultiProx",
    "NonNegative",
    "NuclearNorm",
    "ProxOperator",
    "Shifted",
    "Simplex",
    "Zero",
]
