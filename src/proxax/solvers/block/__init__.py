"""Block-constrained splitting solvers (ADMM)."""

from __future__ import annotations

from proxax.solvers.block import admm

ADMM = admm.ADMM

__all__ = ["ADMM"]
