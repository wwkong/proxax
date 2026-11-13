"""Proximal operators and PyTree combinators."""

from __future__ import annotations

from proxax.prox import base
from proxax.prox import combinators
from proxax.prox import folded_concave
from proxax.prox import indicators
from proxax.prox import norms

Box = indicators.Box
BoxL1 = indicators.BoxL1
ElasticNet = norms.ElasticNet
Fantope = indicators.Fantope
L1 = norms.L1
L1Ball = indicators.L1Ball
L2 = norms.L2
L21 = norms.L21
L2Ball = indicators.L2Ball
L2Squared = norms.L2Squared
LInf = norms.LInf
LInfBall = indicators.LInfBall
MultiProx = combinators.MultiProx
NonNegative = indicators.NonNegative
NuclearNorm = norms.NuclearNorm
ProxOperator = base.ProxOperator
Shifted = base.Shifted
Simplex = indicators.Simplex
TV1D = norms.TV1D
Zero = base.Zero
capped_l1_decomposition = folded_concave.capped_l1_decomposition
logsum_decomposition = folded_concave.logsum_decomposition
mcp_decomposition = folded_concave.mcp_decomposition
scad_decomposition = folded_concave.scad_decomposition

__all__ = [
    "Box",
    "BoxL1",
    "ElasticNet",
    "Fantope",
    "L1",
    "L1Ball",
    "L2",
    "L21",
    "L2Ball",
    "L2Squared",
    "LInf",
    "LInfBall",
    "MultiProx",
    "NonNegative",
    "NuclearNorm",
    "ProxOperator",
    "Shifted",
    "Simplex",
    "TV1D",
    "Zero",
    "capped_l1_decomposition",
    "logsum_decomposition",
    "mcp_decomposition",
    "scad_decomposition",
]
