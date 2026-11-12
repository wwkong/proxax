"""Optax-compatible gradient transformations backed by `proxax`."""

from __future__ import annotations

from proxax.optax_compat import transforms

AIPPTransformState = transforms.AIPPTransformState
APDTransformState = transforms.APDTransformState
IALMTransformState = transforms.IALMTransformState
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
    "AIPPTransformState",
    "APDTransformState",
    "IALMTransformState",
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
