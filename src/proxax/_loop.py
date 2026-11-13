"""Dual-mode loop execution supporting both JAX while_loop and eager Python."""

from __future__ import annotations

import time
import typing
from collections import abc

import jax

from proxax import _typing

StateT = typing.TypeVar("StateT")


def while_loop(
    cond_fun: abc.Callable[[StateT], jax.Array],
    body_fun: abc.Callable[[StateT], StateT],
    init_val: StateT,
    *,
    jit: bool = True,
    callback: _typing.CallbackFn[StateT] | None = None,
    time_limit: float | None = None,
    on_time_limit: abc.Callable[[StateT], StateT] | None = None,
) -> StateT:
  """Executes a stateful loop via `jax.lax.while_loop` or eager Python.

  Args:
    cond_fun: Continuation predicate returning a boolean JAX scalar.
    body_fun: Pure state transition function `state -> next_state`.
    init_val: Initial loop carry PyTree.
    jit: If True, executes `jax.lax.while_loop` (XLA-compiled and
      `vmap`-compatible). If False, runs a Python `while` loop supporting
      Python callbacks and wall-clock time limits.
    callback: Optional Python callback run after each iteration when
      `jit=False`. Returning True stops the loop early.
    time_limit: Optional wall-clock limit in seconds (supported when
      `jit=False`).
    on_time_limit: Optional state updater applied when `time_limit` triggers.

  Returns:
    Final loop state `StateT`.

  Raises:
    ValueError: If `callback` or `time_limit` is supplied when `jit=True`.
  """
  if jit:
    if callback is not None:
      raise ValueError("Python `callback` requires `jit=False`.")
    if time_limit is not None:
      raise ValueError("Wall-clock `time_limit` requires `jit=False`.")
    return jax.lax.while_loop(cond_fun, body_fun, init_val)

  state = init_val
  if callback is not None and callback(state):
    return state
  start_time = time.perf_counter()
  while bool(cond_fun(state)):
    if (
        time_limit is not None
        and (time.perf_counter() - start_time) >= time_limit
    ):
      if on_time_limit is not None:
        state = on_time_limit(state)
      break
    state = body_fun(state)
    if callback is not None and callback(state):
      break
  return state
