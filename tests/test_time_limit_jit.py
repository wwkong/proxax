"""Tests for wall-clock `time_limit` support under `jit=True`."""

from __future__ import annotations

import jax.numpy as jnp

from proxax import px
from proxax import solvers
from proxax.solvers import base


def test_jit_time_limit_stops_loop_early() -> None:
  """`px.solve(..., jit=True, time_limit=...)` stops with TIME_LIMIT_REACHED."""
  n = 200
  a_mat = jnp.eye(n, dtype=jnp.float32) * 1e-4

  def smooth(x: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * jnp.sum((a_mat @ x) ** 2) - jnp.sum(x)

  prob = px.CompositeProblem(
      smooth=smooth,
      proximal=px.prox.Box(lower=jnp.zeros(n), upper=jnp.ones(n)),
      constraints=px.constraints.equality(
          jnp.ones((1, n), dtype=jnp.float32),
          rhs=jnp.array([1.0], dtype=jnp.float32),
      ),
  )
  x0 = jnp.zeros(n, dtype=jnp.float32)
  sol = px.solve(
      prob,
      x0,
      solver=solvers.SProxALM(
          max_iters=100_000, opt_tol=1e-12, feas_tol=1e-12
      ),
      jit=True,
      time_limit=0.001,
  )
  assert sol.status_enum == base.SolveStatus.TIME_LIMIT_REACHED
