"""Top-level `solve` dispatcher for unconstrained and constrained problems."""

from __future__ import annotations

import typing

from proxax import _typing
from proxax import problem as problem_mod
from proxax.solvers import base
from proxax.solvers.constrained import qp_aipp
from proxax.solvers.unconstrained import aipp


def solve(
    problem: problem_mod.CompositeProblem,
    x0: _typing.PyTree,
    *,
    solver: base.Solver | None = None,
    opt_tol: float = 1e-5,
    feas_tol: float = 1e-5,
    jit: bool = True,
    callback: typing.Any = None,
    time_limit: float | None = None,
) -> base.Solution:
  """Solves a `CompositeProblem`, automatically selecting `AIPP` or `QPAIPP`.

  Args:
    problem: Unconstrained or set/cone-constrained `CompositeProblem`.
    x0: Initial primal PyTree iterate.
    solver: Optional explicit solver instance (`AIPP`, `QPAIPP`, etc.). When
      omitted, dispatches to `AIPP(opt_tol=opt_tol)` if unconstrained and
      `QPAIPP(opt_tol=opt_tol, feas_tol=feas_tol)` if constrained.
    opt_tol: Default stationarity tolerance used when `solver` is `None`.
    feas_tol: Default primal feasibility tolerance used when `solver` is `None`.
    jit: If True, executes inner/outer loops via `jax.lax.while_loop` (`jit` and
      `vmap` compatible). If False, runs eager Python loops supporting
      `callback` and `time_limit`.
    callback: Optional per-outer-iteration callback (`jit=False` only).
    time_limit: Optional wall-clock limit in seconds (`jit=False` only).

  Returns:
    A `Solution` PyTree with certified primal `x`, optional `dual`, exact
    stationarity residual `subgrad`, `status`, and `stats`.
  """
  chosen: base.Solver
  if solver is None:
    if problem.is_constrained:
      chosen = qp_aipp.QPAIPP(opt_tol=opt_tol, feas_tol=feas_tol)
    else:
      chosen = aipp.AIPP(opt_tol=opt_tol)
  else:
    chosen = solver

  return chosen.solve(
      problem,
      x0,
      jit=jit,
      callback=callback,
      time_limit=time_limit,
  )
