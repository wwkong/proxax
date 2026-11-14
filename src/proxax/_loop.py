"""Dual-mode loop execution supporting both JAX while_loop and eager Python."""

from __future__ import annotations

import time
import typing
from collections import abc

import jax
import jax.numpy as jnp
import numpy as np

from proxax import _typing

StateT = typing.TypeVar("StateT")

_CLOCK_ORIGIN = time.perf_counter()


def _read_clock_host() -> np.ndarray:
  return np.asarray(time.perf_counter() - _CLOCK_ORIGIN, dtype=np.float32)


def read_clock() -> jax.Array:
  """Reads elapsed process clock seconds inside a JIT-compiled JAX graph."""
  return typing.cast(
      jax.Array,
      jax.experimental.io_callback(
          _read_clock_host,
          jax.eval_shape(lambda: jnp.asarray(0.0, dtype=jnp.float32)),
          ordered=False,
      ),
  )


def start_clock(*, jit: bool) -> float | jax.Array:
  """Captures a wall-clock timestamp for `jit=True` or `jit=False`."""
  return read_clock() if jit else time.perf_counter()


def remaining_time(
    jit: bool,
    time_limit: float | jax.Array | None,
    start_wall: float | jax.Array,
) -> float | None:
  """Computes remaining wall-clock budget for an inner solver stage."""
  if time_limit is None:
    return None
  if jit:
    elapsed = read_clock() - start_wall
    return typing.cast(
        "float | None",
        jnp.maximum(
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(time_limit, dtype=jnp.float32) - elapsed,
        ),
    )
  return max(0.0, float(time_limit) - (time.perf_counter() - float(start_wall)))


def while_loop(
    cond_fun: abc.Callable[[StateT], jax.Array],
    body_fun: abc.Callable[[StateT], StateT],
    init_val: StateT,
    *,
    jit: bool = True,
    callback: _typing.CallbackFn[StateT] | None = None,
    time_limit: float | jax.Array | None = None,
    on_time_limit: abc.Callable[[StateT], StateT] | None = None,
) -> StateT:
  """Executes a stateful loop via `jax.lax.while_loop` or eager Python.

  Args:
    cond_fun: Continuation predicate returning a boolean JAX scalar.
    body_fun: Pure state transition function `state -> next_state`.
    init_val: Initial loop carry PyTree.
    jit: If True, executes `jax.lax.while_loop` (XLA-compiled). If False, runs
      a Python `while` loop.
    callback: Optional Python callback run after each iteration when
      `jit=False` (or via `jax.debug.callback` when supported).
    time_limit: Optional wall-clock limit in seconds (supported for both
      `jit=True` via `io_callback` and `jit=False`).
    on_time_limit: Optional state updater applied when `time_limit` triggers.

  Returns:
    Final loop state `StateT`.

  Raises:
    ValueError: If `callback` is unsupported with `jit=True` or
      `time_limit < 0`.
  """
  if isinstance(time_limit, (int, float)) and time_limit < 0:
    raise ValueError("`time_limit` must be non-negative.")

  if jit:
    if callback is not None and not getattr(
        callback, "_supports_jit_callback", False
    ):
      raise ValueError("Python `callback` requires `jit=False`.")

    if callback is not None:

      def _cb_wrapper(st: StateT) -> None:
        callback(st)

      jax.debug.callback(_cb_wrapper, init_val)

      def _body_with_cb(st: StateT) -> StateT:
        st_next = body_fun(st)
        jax.debug.callback(_cb_wrapper, st_next)
        return st_next

      step_fn: abc.Callable[[StateT], StateT] = _body_with_cb
    else:
      step_fn = body_fun

    if time_limit is not None:
      start_time = read_clock()
      limit_arr = jnp.asarray(time_limit, dtype=jnp.float32)

      def _timed_cond(carry: tuple[StateT, jax.Array]) -> jax.Array:
        st, timed_out = carry
        return (~timed_out) & cond_fun(st)

      def _timed_body(
          carry: tuple[StateT, jax.Array],
      ) -> tuple[StateT, jax.Array]:
        st, _ = carry
        now_out = (read_clock() - start_time) >= limit_arr
        return typing.cast(
            tuple[StateT, jax.Array],
            jax.lax.cond(
                now_out,
                lambda s: (
                    on_time_limit(s) if on_time_limit is not None else s,
                    jnp.asarray(True),
                ),
                lambda s: (step_fn(s), jnp.asarray(False)),
                st,
            ),
        )

      final_state, _ = jax.lax.while_loop(
          _timed_cond,
          _timed_body,
          (init_val, jnp.asarray(False)),
      )
      return final_state

    return jax.lax.while_loop(cond_fun, step_fn, init_val)

  state = init_val
  if callback is not None and callback(state):
    return state
  start_time_py = time.perf_counter()
  while bool(cond_fun(state)):
    if (
        time_limit is not None
        and (time.perf_counter() - start_time_py) >= float(time_limit)
    ):
      if on_time_limit is not None:
        state = on_time_limit(state)
      break
    state = body_fun(state)
    if callback is not None and callback(state):
      break
  return state
