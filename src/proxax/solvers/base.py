"""Common solver status codes, statistics, stopping configs, and solutions."""

from __future__ import annotations

import dataclasses
import enum
import typing

import jax

from proxax import _typing

if typing.TYPE_CHECKING:
  from proxax import problem as problem_mod


class SolveStatus(enum.IntEnum):
  """Integer status codes compatible with `jax.jit` and `jax.vmap`."""

  RUNNING = -1
  CONVERGED = 0
  MAX_ITER_REACHED = 1
  TIME_LIMIT_REACHED = 2
  LINE_SEARCH_FAILED = 3
  NON_FINITE = 4


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class StoppingConfig:
  """Optimality and feasibility termination tolerances.

  Attributes:
    opt_tol: Absolute stationarity tolerance `rho > 0`.
    feas_tol: Absolute primal feasibility tolerance `eta > 0`.
    rel_opt_tol: Optional relative stationarity tolerance against step-0 norm.
    rel_feas_tol: Optional relative feasibility tolerance against step-0 norm.
    norm_kind: Norm used to measure residuals (`'l2'` or `'linf'`).
  """

  opt_tol: float = 1e-5
  feas_tol: float = 1e-5
  rel_opt_tol: float = 0.0
  rel_feas_tol: float = 0.0
  norm_kind: str = "l2"


jax.tree_util.register_dataclass(
    StoppingConfig,
    data_fields=["opt_tol", "feas_tol", "rel_opt_tol", "rel_feas_tol"],
    meta_fields=["norm_kind"],
)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class SolveStats:
  """Iteration and oracle evaluation counters for a solver run."""

  outer_iters: jax.Array
  inner_iters: jax.Array
  smooth_evals: jax.Array
  grad_evals: jax.Array
  prox_evals: jax.Array
  opt_residual: jax.Array
  feas_residual: jax.Array
  obj_val: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Solution:
  """Certified primal-dual solution returned by all `proxax` solvers.

  Attributes:
    x: Certified primal iterate `x_hat in dom(h)`.
    dual: Certified dual multiplier `p_hat` matching `problem.constraints`
      PyTree structure (or `None` if unconstrained).
    subgrad: Certified stationarity residual vector
      `v_hat in grad_f(x_hat) + dh(x_hat) + A^*(p_hat)`.
    status: Scalar `int32` JAX array encoding `SolveStatus`.
    stats: `SolveStats` counters and final residuals.
  """

  x: _typing.PyTree
  dual: _typing.PyTree | None
  subgrad: _typing.PyTree
  status: jax.Array
  stats: SolveStats

  @property
  def status_enum(self) -> SolveStatus:
    """Converts scalar status code to Python `SolveStatus` (eager mode)."""
    return SolveStatus(int(self.status))


@typing.runtime_checkable
class Solver(typing.Protocol):
  """Structural protocol for all `proxax` composite optimization solvers."""

  def solve(
      self,
      problem: problem_mod.CompositeProblem,
      x0: _typing.PyTree,
      *,
      jit: bool = True,
      callback: typing.Any = None,
      time_limit: float | None = None,
  ) -> Solution:
    """Solves `problem` starting from `x0`."""
    ...
