"""Proximal operators and PyTree combinators."""

from __future__ import annotations

from proxax.prox import base
from proxax.prox import combinators
from proxax.prox import exotic
from proxax.prox import folded_concave
from proxax.prox import indicators
from proxax.prox import norms

Box = indicators.Box
BoxL1 = indicators.BoxL1
CappedSimplex = exotic.CappedSimplex
CardinalityBall = exotic.CardinalityBall
ElasticNet = norms.ElasticNet
Fantope = indicators.Fantope
Isotonic = exotic.Isotonic
L0 = exotic.L0
L1 = norms.L1
L1Ball = indicators.L1Ball
L2 = norms.L2
L21 = norms.L21
L2Ball = indicators.L2Ball
L2Squared = norms.L2Squared
LInf = norms.LInf
LInfBall = indicators.LInfBall
LogBarrier = exotic.LogBarrier
MultiProx = combinators.MultiProx
NegEntropy = exotic.NegEntropy
NonNegative = indicators.NonNegative
NuclearNorm = norms.NuclearNorm
OSCAR = exotic.OSCAR
OWL = exotic.OWL
PSDCone = exotic.PSDCone
ProxOperator = base.ProxOperator
SOC = exotic.SOC
Shifted = base.Shifted
Simplex = indicators.Simplex
Spectrahedron = exotic.Spectrahedron
TV1D = norms.TV1D
Zero = base.Zero
capped_l1_decomposition = folded_concave.capped_l1_decomposition
logsum_decomposition = folded_concave.logsum_decomposition
mcp_decomposition = folded_concave.mcp_decomposition
scad_decomposition = folded_concave.scad_decomposition

__all__ = [
    "Box",
    "BoxL1",
    "CappedSimplex",
    "CardinalityBall",
    "ElasticNet",
    "Fantope",
    "Isotonic",
    "L0",
    "L1",
    "L1Ball",
    "L2",
    "L21",
    "L2Ball",
    "L2Squared",
    "LInf",
    "LInfBall",
    "LogBarrier",
    "MultiProx",
    "NegEntropy",
    "NonNegative",
    "NuclearNorm",
    "OSCAR",
    "OWL",
    "PSDCone",
    "ProxOperator",
    "SOC",
    "Shifted",
    "Simplex",
    "Spectrahedron",
    "TV1D",
    "Zero",
    "capped_l1_decomposition",
    "logsum_decomposition",
    "mcp_decomposition",
    "scad_decomposition",
]
