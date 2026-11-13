"""Optax-compatible gradient transformations backed by `proxax`."""

from __future__ import annotations

from proxax.optax_compat import transforms

ACGSubproblemCarry = transforms.ACGSubproblemCarry
AIPPTransformState = transforms.AIPPTransformState
APDTransformState = transforms.APDTransformState
ConstrainedOptaxTracker = transforms.ConstrainedOptaxTracker
IALMTransformState = transforms.IALMTransformState
OptaxTracker = transforms.OptaxTracker
PFAPDSubproblemCarry = transforms.PFAPDSubproblemCarry
ProximalState = transforms.ProximalState
QPAPDTransformState = transforms.QPAPDTransformState
QPTransformState = transforms.QPTransformState
aipp = transforms.aipp
apd = transforms.apd
ialm = transforms.ialm
proximal = transforms.proximal
qp_aipp = transforms.qp_aipp
qp_apd = transforms.qp_apd

__all__ = [
    "ACGSubproblemCarry",
    "AIPPTransformState",
    "APDTransformState",
    "ConstrainedOptaxTracker",
    "IALMTransformState",
    "OptaxTracker",
    "PFAPDSubproblemCarry",
    "ProximalState",
    "QPAPDTransformState",
    "QPTransformState",
    "aipp",
    "apd",
    "ialm",
    "proximal",
    "qp_aipp",
    "qp_apd",
]
