"""Core type aliases and structural protocols for proxax."""

from __future__ import annotations

import typing

import jax

Array: typing.TypeAlias = jax.Array
Scalar: typing.TypeAlias = jax.Array | float
PyTree: typing.TypeAlias = typing.Any
StateT = typing.TypeVar("StateT")
StateContraT = typing.TypeVar("StateContraT", contravariant=True)


@typing.runtime_checkable
class SmoothFn(typing.Protocol):
  """Callable evaluating a scalar smooth objective on a PyTree."""

  def __call__(self, x: PyTree, /) -> jax.Array:
    """Evaluates the smooth function value at `x`."""
    ...


@typing.runtime_checkable
class ValueAndGradFn(typing.Protocol):
  """Callable returning both scalar value and PyTree gradient at `x`."""

  def __call__(self, x: PyTree, /) -> tuple[jax.Array, PyTree]:
    """Evaluates `(f(x), grad_f(x))`."""
    ...


class CallbackFn(typing.Protocol[StateContraT]):
  """Optional eager-mode callback invoked after each outer iteration."""

  def __call__(self, state: StateContraT, /) -> bool | None:
    """Inspects solver state; returning True requests early termination."""
    ...
