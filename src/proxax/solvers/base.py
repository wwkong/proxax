"""Common solver status codes, statistics, trackers, and solutions."""

from __future__ import annotations

import dataclasses
import enum
import typing
from collections import abc

import jax
import jax.numpy as jnp

from proxax import _loop
from proxax import _typing
from proxax import tree

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


def status_code(status: SolveStatus | int) -> jax.Array:
  """Returns a scalar `int32` JAX array for `status`."""
  return jnp.asarray(int(status), dtype=jnp.int32)


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
class OracleCounts:
  """Cumulative smooth-value, gradient, and proximal evaluation counters."""

  smooth_evals: jax.Array
  grad_evals: jax.Array
  prox_evals: jax.Array

  @classmethod
  def zeros(cls) -> OracleCounts:
    """Creates zero-initialized `int32` oracle evaluation counters."""
    zero = jnp.asarray(0, dtype=jnp.int32)
    return cls(smooth_evals=zero, grad_evals=zero, prox_evals=zero)

  @classmethod
  def create(
      cls,
      *,
      smooth: jax.Array | int = 0,
      grad: jax.Array | int = 0,
      prox: jax.Array | int = 0,
  ) -> OracleCounts:
    """Creates `int32` oracle evaluation counters from scalar values."""
    return cls(
        smooth_evals=jnp.asarray(smooth, dtype=jnp.int32),
        grad_evals=jnp.asarray(grad, dtype=jnp.int32),
        prox_evals=jnp.asarray(prox, dtype=jnp.int32),
    )

  def add(
      self,
      *,
      smooth: jax.Array | int = 0,
      grad: jax.Array | int = 0,
      prox: jax.Array | int = 0,
      counts: OracleCounts | None = None,
  ) -> OracleCounts:
    """Returns counters incremented by scalar offsets and/or `counts`."""
    s_next = self.smooth_evals + smooth
    g_next = self.grad_evals + grad
    p_next = self.prox_evals + prox
    if counts is not None:
      s_next = s_next + counts.smooth_evals
      g_next = g_next + counts.grad_evals
      p_next = p_next + counts.prox_evals
    return OracleCounts(
        smooth_evals=s_next,
        grad_evals=g_next,
        prox_evals=p_next,
    )


class OracleCountsMixin:
  """Zero-slot `@property` forwarding mixin for states composing `counts`."""

  __slots__ = ()
  counts: OracleCounts

  @property
  def smooth_evals(self) -> jax.Array:
    """Cumulative smooth function value evaluations."""
    return self.counts.smooth_evals

  @property
  def grad_evals(self) -> jax.Array:
    """Cumulative smooth gradient evaluations."""
    return self.counts.grad_evals

  @property
  def prox_evals(self) -> jax.Array:
    """Cumulative proximal operator evaluations."""
    return self.counts.prox_evals


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


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class SolverTracker:
  """Composable incumbent certificate and counter tracker for solvers.

  Attributes:
    best_x: Best certified primal iterate `x_hat` seen so far.
    best_subgrad: Certified composite subgradient `v_hat` at `best_x`.
    best_opt_res: Stationarity norm `||v_hat||` at `best_x`.
    best_obj: Composite objective value `f(best_x) + h(best_x)`.
    target_opt_tol: Effective stationarity tolerance `opt_tol + rel * res_0`.
    outer_iters: Completed outer iterations.
    inner_iters: Cumulative inner steps / line-search trials.
    counts: Cumulative `OracleCounts`.
    status: Scalar `int32` `SolveStatus` code.
  """

  best_x: _typing.PyTree
  best_subgrad: _typing.PyTree
  best_opt_res: jax.Array
  best_obj: jax.Array
  target_opt_tol: jax.Array
  outer_iters: jax.Array
  inner_iters: jax.Array
  counts: OracleCounts
  status: jax.Array

  @property
  def smooth_evals(self) -> jax.Array:
    """Cumulative smooth function value evaluations."""
    return self.counts.smooth_evals

  @property
  def grad_evals(self) -> jax.Array:
    """Cumulative smooth gradient evaluations."""
    return self.counts.grad_evals

  @property
  def prox_evals(self) -> jax.Array:
    """Cumulative proximal operator evaluations."""
    return self.counts.prox_evals

  @property
  def status_enum(self) -> SolveStatus:
    """Converts scalar status code to Python `SolveStatus` (eager mode)."""
    return SolveStatus(int(self.status))

  @classmethod
  def from_init(
      cls,
      x: _typing.PyTree,
      subgrad: _typing.PyTree,
      opt_res: jax.Array,
      obj: jax.Array,
      *,
      opt_tol: _typing.Scalar,
      rel_opt_tol: _typing.Scalar = 0.0,
      opt_tol_override: _typing.Scalar | None = None,
      smooth_evals: jax.Array | int = 0,
      grad_evals: jax.Array | int = 0,
      prox_evals: jax.Array | int = 0,
      counts: OracleCounts | None = None,
  ) -> SolverTracker:
    """Initializes `SolverTracker` from step-0 certificate quantities."""
    dtype = opt_res.dtype
    if opt_tol_override is not None:
      target_tol = jnp.asarray(opt_tol_override, dtype=dtype)
    else:
      target_tol = jnp.asarray(opt_tol, dtype=dtype) + (
          jnp.asarray(rel_opt_tol, dtype=dtype) * opt_res
      )
    finite0 = jnp.logical_and(tree.all_finite(x), jnp.isfinite(opt_res))
    status0 = jnp.where(
        jnp.logical_not(finite0),
        status_code(SolveStatus.NON_FINITE),
        jnp.where(
            opt_res <= target_tol,
            status_code(SolveStatus.CONVERGED),
            status_code(SolveStatus.RUNNING),
        ),
    )
    init_counts = OracleCounts.create(
        smooth=smooth_evals,
        grad=grad_evals,
        prox=prox_evals,
    )
    if counts is not None:
      init_counts = init_counts.add(counts=counts)
    zero_i32 = jnp.asarray(0, dtype=jnp.int32)
    return cls(
        best_x=x,
        best_subgrad=subgrad,
        best_opt_res=opt_res,
        best_obj=obj,
        target_opt_tol=target_tol,
        outer_iters=zero_i32,
        inner_iters=zero_i32,
        counts=init_counts,
        status=status0,
    )

  def update_incumbent(
      self,
      x: _typing.PyTree,
      subgrad: _typing.PyTree,
      opt_res: jax.Array,
      obj: jax.Array,
  ) -> SolverTracker:
    """Updates best certified `(x, subgrad, opt_res, obj)` when improved."""
    cand_finite = jnp.logical_and(tree.all_finite(x), jnp.isfinite(opt_res))
    is_better = jnp.logical_and(cand_finite, opt_res <= self.best_opt_res)
    best_x = jax.tree.map(
        lambda cx, bx: jnp.where(is_better, cx, bx),
        x,
        self.best_x,
    )
    best_v = jax.tree.map(
        lambda cv, bv: jnp.where(is_better, cv, bv),
        subgrad,
        self.best_subgrad,
    )
    best_res = jnp.where(is_better, opt_res, self.best_opt_res)
    best_obj = jnp.where(is_better, obj, self.best_obj)
    return dataclasses.replace(
        self,
        best_x=best_x,
        best_subgrad=best_v,
        best_opt_res=best_res,
        best_obj=best_obj,
    )

  def advance(
      self,
      *,
      outer_inc: jax.Array | int = 1,
      inner_inc: jax.Array | int = 0,
      smooth_inc: jax.Array | int = 0,
      grad_inc: jax.Array | int = 0,
      prox_inc: jax.Array | int = 0,
      counts_inc: OracleCounts | None = None,
      max_iters: int | jax.Array | None = None,
      check_converged: bool = True,
      line_search_failed: jax.Array | bool = False,
      finite_ok: jax.Array | None = None,
  ) -> SolverTracker:
    """Increments counters and computes next termination `status`."""
    next_outer = self.outer_iters + outer_inc
    next_inner = self.inner_iters + inner_inc
    next_counts = self.counts.add(
        smooth=smooth_inc,
        grad=grad_inc,
        prox=prox_inc,
        counts=counts_inc,
    )
    is_finite = (
        jnp.logical_and(
            tree.all_finite(self.best_x),
            jnp.isfinite(self.best_opt_res),
        )
        if finite_ok is None
        else finite_ok
    )
    ls_fail = jnp.asarray(line_search_failed, dtype=jnp.bool_)
    conv = (
        (self.best_opt_res <= self.target_opt_tol)
        if check_converged
        else jnp.asarray(False, dtype=jnp.bool_)
    )
    hit_max = (
        (next_outer >= max_iters)
        if max_iters is not None
        else jnp.asarray(False, dtype=jnp.bool_)
    )
    next_status = jnp.where(
        jnp.logical_not(is_finite),
        status_code(SolveStatus.NON_FINITE),
        jnp.where(
            ls_fail,
            status_code(SolveStatus.LINE_SEARCH_FAILED),
            jnp.where(
                conv,
                status_code(SolveStatus.CONVERGED),
                jnp.where(
                    hit_max,
                    status_code(SolveStatus.MAX_ITER_REACHED),
                    status_code(SolveStatus.RUNNING),
                ),
            ),
        ),
    )
    return dataclasses.replace(
        self,
        outer_iters=next_outer,
        inner_iters=next_inner,
        counts=next_counts,
        status=next_status,
    )

  def with_status(self, status: SolveStatus | jax.Array) -> SolverTracker:
    """Returns a copy with `status` replaced."""
    code = (
        status
        if isinstance(status, jax.Array)
        else status_code(status)
    )
    return dataclasses.replace(self, status=code)

  def to_solution(self) -> Solution:
    """Packs tracked incumbent certificate and counters into `Solution`."""
    stats = SolveStats(
        outer_iters=self.outer_iters,
        inner_iters=self.inner_iters,
        smooth_evals=self.counts.smooth_evals,
        grad_evals=self.counts.grad_evals,
        prox_evals=self.counts.prox_evals,
        opt_residual=self.best_opt_res,
        feas_residual=jnp.asarray(0.0, dtype=self.best_opt_res.dtype),
        obj_val=self.best_obj,
    )
    return Solution(
        x=self.best_x,
        dual=None,
        subgrad=self.best_subgrad,
        status=self.status,
        stats=stats,
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ConstrainedTracker:
  """Composable KKT certificate and counter tracker for constrained solvers.

  Attributes:
    best_x: Best certified primal iterate `x_hat` seen so far.
    best_dual: Certified dual multiplier `p_hat` at `best_x`.
    best_subgrad: Certified Lagrangian subgradient `v_hat` at `best_x`.
    best_opt_res: Lagrangian stationarity norm `||v_hat||` at `best_x`.
    best_feas_res: Primal feasibility / complementarity residual at `best_x`.
    best_kkt_score: Normalized max KKT violation score at `best_x`.
    best_obj: Primal composite objective `f(best_x) + h(best_x)`.
    target_opt_tol: Effective stationarity tolerance.
    target_feas_tol: Effective feasibility tolerance.
    outer_iters: Completed outer penalty / AL stages.
    inner_iters: Cumulative inner iterations.
    counts: Cumulative `OracleCounts`.
    status: Scalar `int32` `SolveStatus` code.
  """

  best_x: _typing.PyTree
  best_dual: _typing.PyTree
  best_subgrad: _typing.PyTree
  best_opt_res: jax.Array
  best_feas_res: jax.Array
  best_kkt_score: jax.Array
  best_obj: jax.Array
  target_opt_tol: jax.Array
  target_feas_tol: jax.Array
  outer_iters: jax.Array
  inner_iters: jax.Array
  counts: OracleCounts
  status: jax.Array

  @property
  def smooth_evals(self) -> jax.Array:
    """Cumulative smooth function value evaluations."""
    return self.counts.smooth_evals

  @property
  def grad_evals(self) -> jax.Array:
    """Cumulative smooth gradient evaluations."""
    return self.counts.grad_evals

  @property
  def prox_evals(self) -> jax.Array:
    """Cumulative proximal operator evaluations."""
    return self.counts.prox_evals

  @property
  def status_enum(self) -> SolveStatus:
    """Converts scalar status code to Python `SolveStatus` (eager mode)."""
    return SolveStatus(int(self.status))

  @classmethod
  def from_init(
      cls,
      x: _typing.PyTree,
      dual: _typing.PyTree,
      subgrad: _typing.PyTree,
      opt_res: jax.Array,
      feas_res: jax.Array,
      obj: jax.Array,
      *,
      opt_tol: _typing.Scalar,
      feas_tol: _typing.Scalar,
      rel_opt_tol: _typing.Scalar = 0.0,
      rel_feas_tol: _typing.Scalar = 0.0,
      smooth_evals: jax.Array | int = 0,
      grad_evals: jax.Array | int = 0,
      prox_evals: jax.Array | int = 0,
      counts: OracleCounts | None = None,
  ) -> ConstrainedTracker:
    """Initializes `ConstrainedTracker` from step-0 KKT quantities."""
    dtype = opt_res.dtype
    target_opt = jnp.asarray(opt_tol, dtype=dtype) + (
        jnp.asarray(rel_opt_tol, dtype=dtype) * opt_res
    )
    target_feas = jnp.asarray(feas_tol, dtype=dtype) + (
        jnp.asarray(rel_feas_tol, dtype=dtype) * feas_res
    )
    tiny = jnp.asarray(1e-30, dtype=dtype)
    score0 = jnp.maximum(
        opt_res / jnp.maximum(target_opt, tiny),
        feas_res / jnp.maximum(target_feas, tiny),
    )
    finite0 = jnp.logical_and(
        tree.all_finite(x),
        jnp.logical_and(jnp.isfinite(opt_res), jnp.isfinite(feas_res)),
    )
    conv0 = jnp.logical_and(opt_res <= target_opt, feas_res <= target_feas)
    status0 = jnp.where(
        jnp.logical_not(finite0),
        status_code(SolveStatus.NON_FINITE),
        jnp.where(
            conv0,
            status_code(SolveStatus.CONVERGED),
            status_code(SolveStatus.RUNNING),
        ),
    )
    init_counts = OracleCounts.create(
        smooth=smooth_evals,
        grad=grad_evals,
        prox=prox_evals,
    )
    if counts is not None:
      init_counts = init_counts.add(counts=counts)
    zero_i32 = jnp.asarray(0, dtype=jnp.int32)
    return cls(
        best_x=x,
        best_dual=dual,
        best_subgrad=subgrad,
        best_opt_res=opt_res,
        best_feas_res=feas_res,
        best_kkt_score=score0,
        best_obj=obj,
        target_opt_tol=target_opt,
        target_feas_tol=target_feas,
        outer_iters=zero_i32,
        inner_iters=zero_i32,
        counts=init_counts,
        status=status0,
    )

  def update_incumbent(
      self,
      x: _typing.PyTree,
      dual: _typing.PyTree,
      subgrad: _typing.PyTree,
      opt_res: jax.Array,
      feas_res: jax.Array,
      obj: jax.Array,
  ) -> ConstrainedTracker:
    """Updates best KKT triple `(x, dual, subgrad)` by normalized KKT score."""
    dtype = self.best_opt_res.dtype
    tiny = jnp.asarray(1e-30, dtype=dtype)
    cand_score = jnp.maximum(
        opt_res / jnp.maximum(self.target_opt_tol, tiny),
        feas_res / jnp.maximum(self.target_feas_tol, tiny),
    )
    cand_finite = jnp.logical_and(
        tree.all_finite(x),
        jnp.logical_and(jnp.isfinite(opt_res), jnp.isfinite(feas_res)),
    )
    is_better = jnp.logical_and(cand_finite, cand_score <= self.best_kkt_score)
    best_x = jax.tree.map(
        lambda cx, bx: jnp.where(is_better, cx, bx),
        x,
        self.best_x,
    )
    best_p = jax.tree.map(
        lambda cp, bp: jnp.where(is_better, cp, bp),
        dual,
        self.best_dual,
    )
    best_v = jax.tree.map(
        lambda cv, bv: jnp.where(is_better, cv, bv),
        subgrad,
        self.best_subgrad,
    )
    best_opt = jnp.where(is_better, opt_res, self.best_opt_res)
    best_feas = jnp.where(is_better, feas_res, self.best_feas_res)
    best_score = jnp.where(is_better, cand_score, self.best_kkt_score)
    best_obj = jnp.where(is_better, obj, self.best_obj)
    return dataclasses.replace(
        self,
        best_x=best_x,
        best_dual=best_p,
        best_subgrad=best_v,
        best_opt_res=best_opt,
        best_feas_res=best_feas,
        best_kkt_score=best_score,
        best_obj=best_obj,
    )

  def advance(
      self,
      *,
      outer_inc: jax.Array | int = 1,
      inner_inc: jax.Array | int = 0,
      smooth_inc: jax.Array | int = 0,
      grad_inc: jax.Array | int = 0,
      prox_inc: jax.Array | int = 0,
      counts_inc: OracleCounts | None = None,
      max_iters: int | jax.Array | None = None,
      check_converged: bool = True,
      sub_status: jax.Array | None = None,
      line_search_failed: jax.Array | bool = False,
      finite_ok: jax.Array | None = None,
  ) -> ConstrainedTracker:
    """Increments counters and computes next KKT termination `status`."""
    next_outer = self.outer_iters + outer_inc
    next_inner = self.inner_iters + inner_inc
    next_counts = self.counts.add(
        smooth=smooth_inc,
        grad=grad_inc,
        prox=prox_inc,
        counts=counts_inc,
    )
    is_finite = (
        jnp.logical_and(
            tree.all_finite(self.best_x),
            jnp.logical_and(
                jnp.isfinite(self.best_opt_res),
                jnp.isfinite(self.best_feas_res),
            ),
        )
        if finite_ok is None
        else finite_ok
    )
    timeout_code = status_code(SolveStatus.TIME_LIMIT_REACHED)
    ls_code = status_code(SolveStatus.LINE_SEARCH_FAILED)
    timed_out = (
        (sub_status == timeout_code)
        if sub_status is not None
        else jnp.asarray(False, dtype=jnp.bool_)
    )
    ls_fail = jnp.asarray(line_search_failed, dtype=jnp.bool_)
    if sub_status is not None:
      ls_fail = jnp.logical_or(ls_fail, sub_status == ls_code)
    kkt_conv = (
        jnp.logical_and(
            self.best_opt_res <= self.target_opt_tol,
            self.best_feas_res <= self.target_feas_tol,
        )
        if check_converged
        else jnp.asarray(False, dtype=jnp.bool_)
    )
    hit_max = (
        (next_outer >= max_iters)
        if max_iters is not None
        else jnp.asarray(False, dtype=jnp.bool_)
    )
    next_status = jnp.where(
        jnp.logical_not(is_finite),
        status_code(SolveStatus.NON_FINITE),
        jnp.where(
            timed_out,
            timeout_code,
            jnp.where(
                ls_fail,
                ls_code,
                jnp.where(
                    kkt_conv,
                    status_code(SolveStatus.CONVERGED),
                    jnp.where(
                        hit_max,
                        status_code(SolveStatus.MAX_ITER_REACHED),
                        status_code(SolveStatus.RUNNING),
                    ),
                ),
            ),
        ),
    )
    return dataclasses.replace(
        self,
        outer_iters=next_outer,
        inner_iters=next_inner,
        counts=next_counts,
        status=next_status,
    )

  def with_status(self, status: SolveStatus | jax.Array) -> ConstrainedTracker:
    """Returns a copy with `status` replaced."""
    code = (
        status
        if isinstance(status, jax.Array)
        else status_code(status)
    )
    return dataclasses.replace(self, status=code)

  def to_solution(self) -> Solution:
    """Packs tracked KKT certificate and counters into `Solution`."""
    stats = SolveStats(
        outer_iters=self.outer_iters,
        inner_iters=self.inner_iters,
        smooth_evals=self.counts.smooth_evals,
        grad_evals=self.counts.grad_evals,
        prox_evals=self.counts.prox_evals,
        opt_residual=self.best_opt_res,
        feas_residual=self.best_feas_res,
        obj_val=self.best_obj,
    )
    return Solution(
        x=self.best_x,
        dual=self.best_dual,
        subgrad=self.best_subgrad,
        status=self.status,
        stats=stats,
    )


class SolverTrackerMixin:
  """Zero-slot `@property` forwarding mixin for states composing `tracker`."""

  __slots__ = ()
  tracker: SolverTracker | ConstrainedTracker

  @property
  def best_x(self) -> _typing.PyTree:
    """Best certified primal iterate `x_hat`."""
    return self.tracker.best_x

  @property
  def best_subgrad(self) -> _typing.PyTree:
    """Certified subgradient `v_hat` at `best_x`."""
    return self.tracker.best_subgrad

  @property
  def best_opt_res(self) -> jax.Array:
    """Stationarity residual norm `||v_hat||` at `best_x`."""
    return self.tracker.best_opt_res

  @property
  def best_obj(self) -> jax.Array:
    """Composite objective value at `best_x`."""
    return self.tracker.best_obj

  @property
  def target_opt_tol(self) -> jax.Array:
    """Effective target stationarity tolerance."""
    return self.tracker.target_opt_tol

  @property
  def outer_iters(self) -> jax.Array:
    """Completed outer solver iterations."""
    return self.tracker.outer_iters

  @property
  def inner_iters(self) -> jax.Array:
    """Cumulative inner subproblem / line-search steps."""
    return self.tracker.inner_iters

  @property
  def counts(self) -> OracleCounts:
    """Cumulative `OracleCounts`."""
    return self.tracker.counts

  @property
  def smooth_evals(self) -> jax.Array:
    """Cumulative smooth value evaluations."""
    return self.tracker.counts.smooth_evals

  @property
  def grad_evals(self) -> jax.Array:
    """Cumulative smooth gradient evaluations."""
    return self.tracker.counts.grad_evals

  @property
  def prox_evals(self) -> jax.Array:
    """Cumulative proximal operator evaluations."""
    return self.tracker.counts.prox_evals

  @property
  def status(self) -> jax.Array:
    """Scalar `int32` `SolveStatus` code."""
    return self.tracker.status

  @property
  def status_enum(self) -> SolveStatus:
    """Python `SolveStatus` enum (eager mode)."""
    return self.tracker.status_enum

  def to_solution(self) -> Solution:
    """Converts the tracked solver state into a `Solution`."""
    return self.tracker.to_solution()

  def with_status(self, status: SolveStatus | jax.Array) -> typing.Self:
    """Returns a copy of this state with `tracker.status` updated."""
    cls_any: typing.Any = self
    new_obj = dataclasses.replace(
        cls_any,
        tracker=self.tracker.with_status(status),
    )
    return typing.cast(typing.Self, new_obj)


class ConstrainedTrackerMixin(SolverTrackerMixin):
  """Zero-slot `@property` forwarding mixin for constrained solver states."""

  __slots__ = ()
  tracker: ConstrainedTracker

  @property
  def best_dual(self) -> _typing.PyTree:
    """Certified dual multiplier `p_hat` at `best_x`."""
    return self.tracker.best_dual

  @property
  def best_feas_res(self) -> jax.Array:
    """Primal feasibility / complementarity residual at `best_x`."""
    return self.tracker.best_feas_res

  @property
  def best_kkt_score(self) -> jax.Array:
    """Normalized KKT violation score at `best_x`."""
    return self.tracker.best_kkt_score

  @property
  def target_feas_tol(self) -> jax.Array:
    """Effective target feasibility tolerance."""
    return self.tracker.target_feas_tol


TrackedStateT = typing.TypeVar("TrackedStateT", bound=SolverTrackerMixin)


def run_solver_loop(
    body_fun: abc.Callable[[TrackedStateT], TrackedStateT],
    init_state: TrackedStateT,
    *,
    jit: bool = True,
    callback: _typing.CallbackFn[TrackedStateT] | None = None,
    time_limit: float | None = None,
) -> TrackedStateT:
  """Runs an outer solver loop until `state.tracker.status != RUNNING`."""
  running_code = status_code(SolveStatus.RUNNING)

  def _cond(st: TrackedStateT) -> jax.Array:
    return st.tracker.status == running_code

  def _on_timeout(st: TrackedStateT) -> TrackedStateT:
    return st.with_status(SolveStatus.TIME_LIMIT_REACHED)

  return _loop.while_loop(
      _cond,
      body_fun,
      init_state,
      jit=jit,
      callback=callback,
      time_limit=time_limit,
      on_time_limit=_on_timeout,
  )


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
