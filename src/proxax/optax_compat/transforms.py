"""Optax gradient transformations for `proxax` operators and solvers."""

from __future__ import annotations

from proxax.optax_compat._subproblems import ACGSubproblemCarry
from proxax.optax_compat._subproblems import ACGSubproblemMixin
from proxax.optax_compat._subproblems import ConstrainedOptaxTrackerMixin
from proxax.optax_compat._subproblems import OptaxTrackerMixin
from proxax.optax_compat._subproblems import PFAPDSubproblemCarry
from proxax.optax_compat._subproblems import PFAPDSubproblemMixin
from proxax.optax_compat._trackers import ConstrainedOptaxTracker
from proxax.optax_compat._trackers import OptaxTracker
from proxax.optax_compat._trackers import ProximalState
from proxax.optax_compat._trackers import StepSizeOrSchedule
from proxax.optax_compat._trackers import proximal
from proxax.optax_compat.aipp import AIPPTransformState
from proxax.optax_compat.aipp import aipp
from proxax.optax_compat.apd import APDTransformState
from proxax.optax_compat.apd import apd
from proxax.optax_compat.constrained import IALMTransformState
from proxax.optax_compat.constrained import QPAPDTransformState
from proxax.optax_compat.constrained import QPTransformState
from proxax.optax_compat.constrained import ialm
from proxax.optax_compat.constrained import qp_aipp
from proxax.optax_compat.constrained import qp_apd

__all__ = [
    "ACGSubproblemCarry",
    "ACGSubproblemMixin",
    "AIPPTransformState",
    "APDTransformState",
    "ConstrainedOptaxTracker",
    "ConstrainedOptaxTrackerMixin",
    "IALMTransformState",
    "OptaxTracker",
    "OptaxTrackerMixin",
    "PFAPDSubproblemCarry",
    "PFAPDSubproblemMixin",
    "ProximalState",
    "QPAPDTransformState",
    "QPTransformState",
    "StepSizeOrSchedule",
    "aipp",
    "apd",
    "ialm",
    "proximal",
    "qp_aipp",
    "qp_apd",
]
