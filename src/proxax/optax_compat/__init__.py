"""Optax-compatible gradient transformations backed by `proxax`."""

from __future__ import annotations

from proxax.optax_compat import transforms

AIPPTransformState = transforms.AIPPTransformState
APDTransformState = transforms.APDTransformState
ProximalState = transforms.ProximalState
QPAPDTransformState = transforms.QPAPDTransformState
QPTransformState = transforms.QPTransformState
aipp = transforms.aipp
apd = transforms.apd
proximal = transforms.proximal
qp_aipp = transforms.qp_aipp
qp_apd = transforms.qp_apd

__all__ = [
    "AIPPTransformState",
    "APDTransformState",
    "ProximalState",
    "QPAPDTransformState",
    "QPTransformState",
    "aipp",
    "apd",
    "proximal",
    "qp_aipp",
    "qp_apd",
]
