"""Composite oracle abstractions and parameterized transformations."""

from __future__ import annotations

from proxax.oracle import block
from proxax.oracle import composite
from proxax.oracle import minmax
from proxax.oracle import spectral
from proxax.oracle import terms

BlockProblem = block.BlockProblem
BlockTerm = block.BlockTerm
CompositeOracle = composite.CompositeOracle
CompositeOracleProtocol = composite.CompositeOracleProtocol
MinMaxOracle = minmax.MinMaxOracle
MinMaxProblem = minmax.MinMaxProblem
PenalizedOracle = composite.PenalizedOracle
ProxRegularizedOracle = composite.ProxRegularizedOracle
RescaledProxOracle = composite.RescaledProxOracle
SmoothObjective = terms.SmoothObjective
SmoothTerm = terms.SmoothTerm
SpectralOracle = spectral.SpectralOracle
SpectralProblem = spectral.SpectralProblem
SpectralRefinementResult = spectral.SpectralRefinementResult
VectorSpectralSubOracle = spectral.VectorSpectralSubOracle
ZeroSmooth = terms.ZeroSmooth
as_smooth_term = terms.as_smooth_term
as_spectral_oracle = spectral.as_spectral_oracle

__all__ = [
    "BlockProblem",
    "BlockTerm",
    "CompositeOracle",
    "CompositeOracleProtocol",
    "MinMaxOracle",
    "MinMaxProblem",
    "PenalizedOracle",
    "ProxRegularizedOracle",
    "RescaledProxOracle",
    "SmoothObjective",
    "SmoothTerm",
    "SpectralOracle",
    "SpectralProblem",
    "SpectralRefinementResult",
    "VectorSpectralSubOracle",
    "ZeroSmooth",
    "as_smooth_term",
    "as_spectral_oracle",
]
