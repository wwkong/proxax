"""Public type aliases and structural protocols (`px.typing`)."""

from __future__ import annotations

from proxax import _typing

Array = _typing.Array
CallbackFn = _typing.CallbackFn
PyTree = _typing.PyTree
Scalar = _typing.Scalar
SmoothFn = _typing.SmoothFn
ValueAndGradFn = _typing.ValueAndGradFn

__all__ = [
    "Array",
    "CallbackFn",
    "PyTree",
    "Scalar",
    "SmoothFn",
    "ValueAndGradFn",
]
