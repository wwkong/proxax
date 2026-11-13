"""Block-constrained splitting solvers (ADMM)."""

from __future__ import annotations

from proxax.solvers.block import admm
from proxax.solvers.block import dp_admm
from proxax.solvers.block import p_adapt_admm

ADMM = admm.ADMM
DPADMM = dp_admm.DPADMM
PAdaptADMM = p_adapt_admm.PAdaptADMM

__all__ = ["ADMM", "DPADMM", "PAdaptADMM"]
