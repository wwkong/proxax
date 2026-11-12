"""Optax-compatible gradient transformations backed by `proxax`."""

from __future__ import annotations

from proxax.optax_compat import transforms

AIPPTransformState = transforms.AIPPTransformState
ProximalState = transforms.ProximalState
QPTransformState = transforms.QPTransformState
aipp = transforms.aipp
proximal = transforms.proximal
qp_aipp = transforms.qp_aipp

__all__ = [
    "AIPPTransformState",
    "ProximalState",
    "QPTransformState",
    "aipp",
    "proximal",
    "qp_aipp",
]
