"""Common solver status codes, statistics, trackers, and solutions."""

from __future__ import annotations

import dataclasses
import enum
import time
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

  def __repr__(self) -> str:
    """Returns a compact string representation of evaluation counts."""
    return (
        "OracleCounts("
        f"smooth={_fmt_int(self.smooth_evals)}, "
        f"grad={_fmt_int(self.grad_evals)}, "
        f"prox={_fmt_int(self.prox_evals)})"
    )


def _fmt_int(val: jax.Array) -> str:
  """Formats a scalar integer array, staying safe under `jax.jit` tracing."""
  if isinstance(val, jax.core.Tracer):
    return "<traced>"
  try:
    return str(int(val))
  except (TypeError, ValueError):
    return repr(val)


def _fmt_float(val: jax.Array, spec: str = ".6e") -> str:
  """Formats a scalar float array, staying safe under `jax.jit` tracing."""
  if isinstance(val, jax.core.Tracer):
    return "<traced>"
  try:
    return f"{float(val):{spec}}"
  except (TypeError, ValueError):
    return repr(val)


def _fmt_status(status: jax.Array) -> str:
  """Formats a scalar `SolveStatus` code, staying safe under tracing."""
  if isinstance(status, jax.core.Tracer):
    return "<traced>"
  try:
    return SolveStatus(int(status)).name
  except (TypeError, ValueError):
    return repr(status)


def _fmt_pytree(value: _typing.PyTree | None) -> str:
  """Summarizes a PyTree's shape/dtype without printing raw array values."""
  if value is None:
    return "None"
  leaves = jax.tree_util.tree_leaves(value)
  if not leaves:
    return "PyTree(0 leaves)"
  if len(leaves) == 1:
    leaf = leaves[0]
    if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
      dims = ",".join(str(d) for d in leaf.shape)
      return f"{leaf.dtype}[{dims}]"
    return type(leaf).__name__
  total = sum(int(leaf.size) for leaf in leaves if hasattr(leaf, "size"))
  return f"PyTree({len(leaves)} leaves, {total} elems)"


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


def _init_status(finite: jax.Array, converged: jax.Array) -> jax.Array:
  """Computes initial `SolveStatus` code from finiteness and convergence."""
  return jnp.where(
      jnp.logical_not(finite),
      status_code(SolveStatus.NON_FINITE),
      jnp.where(
          converged,
          status_code(SolveStatus.CONVERGED),
          status_code(SolveStatus.RUNNING),
      ),
  )


def _build_init_counts(
    smooth: jax.Array | int,
    grad: jax.Array | int,
    prox: jax.Array | int,
    counts: OracleCounts | None,
) -> OracleCounts:
  """Builds initial `OracleCounts` from scalar counts plus optional bundle."""
  init_counts = OracleCounts.create(smooth=smooth, grad=grad, prox=prox)
  return init_counts.add(counts=counts) if counts is not None else init_counts


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

  def __repr__(self) -> str:
    """Returns a concise summary of solver statistics."""
    return (
        "SolveStats(\n"
        f"  iters={_fmt_int(self.outer_iters)} "
        f"(inner={_fmt_int(self.inner_iters)}),\n"
        f"  obj={_fmt_float(self.obj_val)},\n"
        f"  opt_res={_fmt_float(self.opt_residual, '.3e')},\n"
        f"  feas_res={_fmt_float(self.feas_residual, '.3e')},\n"
        f"  evals=(smooth={_fmt_int(self.smooth_evals)}, "
        f"grad={_fmt_int(self.grad_evals)}, "
        f"prox={_fmt_int(self.prox_evals)}),\n"
        ")"
    )


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
    history: Optional `HistoryTracker` recorded when `track_history=True`.
  """

  x: _typing.PyTree
  dual: _typing.PyTree | None
  subgrad: _typing.PyTree
  status: jax.Array
  stats: SolveStats
  history: HistoryTracker | None = dataclasses.field(default=None)

  @property
  def status_enum(self) -> SolveStatus:
    """Converts scalar status code to Python `SolveStatus` (eager mode)."""
    return SolveStatus(int(self.status))

  def plot(
      self,
      *,
      x_axis: typing.Literal["iter", "time", "evals"] = "iter",
      title: str | None = None,
      show_tol: bool = True,
      alpha: float = 0.75,
      style: str | None = "mocha",
      ax: typing.Any = None,
  ) -> tuple[typing.Any, list[typing.Any]]:
    """Plots recorded convergence curves if `self.history` is present."""
    if self.history is None:
      raise ValueError(
          "Solution has no recorded history. Re-run with "
          "`proxax.solve(..., jit=False, track_history=True)` or pass a "
          "`HistoryTracker` via `callback=`."
      )
    return self.history.plot(
        x_axis=x_axis,
        title=title,
        show_tol=show_tol,
        alpha=alpha,
        style=style,
        ax=ax,
    )

  def plot_spectrum(
      self,
      *,
      title: str | None = None,
      log_y: bool = True,
      alpha: float = 0.75,
      style: str | None = "mocha",
      ax: typing.Any = None,
  ) -> tuple[typing.Any, typing.Any]:
    """Plots the singular-value spectrum of a 2D matrix solution `self.x`."""
    from proxax import plotting  # noqa: PLC0415

    return plotting.plot_spectrum(
        self.x,
        title=title,
        log_y=log_y,
        alpha=alpha,
        style=style,
        ax=ax,
    )

  def __repr__(self) -> str:
    """Returns a concise summary without dumping full array payloads."""
    st = self.stats
    return (
        "Solution(\n"
        f"  status={_fmt_status(self.status)},\n"
        f"  obj={_fmt_float(st.obj_val)},\n"
        f"  opt_res={_fmt_float(st.opt_residual, '.3e')},\n"
        f"  feas_res={_fmt_float(st.feas_residual, '.3e')},\n"
        f"  iters={_fmt_int(st.outer_iters)} "
        f"(inner={_fmt_int(st.inner_iters)}),\n"
        f"  evals=(smooth={_fmt_int(st.smooth_evals)}, "
        f"grad={_fmt_int(st.grad_evals)}, "
        f"prox={_fmt_int(st.prox_evals)}),\n"
        f"  x={_fmt_pytree(self.x)},\n"
        f"  dual={_fmt_pytree(self.dual)},\n"
        ")"
    )


jax.tree_util.register_dataclass(
    Solution,
    data_fields=["x", "dual", "subgrad", "status", "stats"],
    meta_fields=["history"],
)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class SolverTracker(OracleCountsMixin):
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
    status0 = _init_status(finite0, opt_res <= target_tol)
    init_counts = _build_init_counts(
        smooth_evals, grad_evals, prox_evals, counts
    )
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
class ConstrainedTracker(OracleCountsMixin):
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
    status0 = _init_status(finite0, conv0)
    init_counts = _build_init_counts(
        smooth_evals, grad_evals, prox_evals, counts
    )
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


class HistoryTracker:
  """Records per-iteration objective values, residuals, and tolerances.

  Designed for use as an eager-mode `callback` in `proxax.solve(..., jit=False,
  callback=tracker)` or direct `solver.solve(..., jit=False, callback=tracker)`.

  Attributes:
    verbose: If `True`, prints a formatted progress table while solving.
    print_every: Row print interval when `verbose=True`.
    outer_iters: Recorded outer iteration counts.
    inner_iters: Recorded cumulative inner iteration counts.
    obj_vals: Recorded best composite objective values.
    opt_residuals: Recorded optimality residuals.
    feas_residuals: Recorded feasibility residuals (`0.0` if unconstrained).
    opt_tols: Recorded effective optimality tolerances.
    feas_tols: Recorded effective feasibility tolerances (`0.0` if
      unconstrained).
    smooth_evals: Recorded cumulative smooth evaluations.
    grad_evals: Recorded cumulative gradient evaluations.
    prox_evals: Recorded cumulative proximal evaluations.
    elapsed_times: Wall-clock seconds since tracker creation.
  """

  def __init__(
      self,
      *,
      verbose: bool = False,
      print_every: int = 1,
  ) -> None:
    """Initializes an empty history tracker."""
    if print_every < 1:
      raise ValueError("`print_every` must be >= 1.")
    self.verbose = verbose
    self.print_every = print_every
    self.outer_iters: list[int] = []
    self.inner_iters: list[int] = []
    self.obj_vals: list[float] = []
    self.opt_residuals: list[float] = []
    self.feas_residuals: list[float] = []
    self.opt_tols: list[float] = []
    self.feas_tols: list[float] = []
    self.smooth_evals: list[int] = []
    self.grad_evals: list[int] = []
    self.prox_evals: list[int] = []
    self.elapsed_times: list[float] = []
    self._t0 = time.perf_counter()
    self._printed_header = False

  def __call__(
      self,
      state: SolverTrackerMixin | SolverTracker | ConstrainedTracker,
  ) -> bool:
    """Appends metrics from `state` and optionally prints a progress row."""
    outer = int(state.outer_iters)
    inner = int(state.inner_iters)
    obj = float(state.best_obj)
    opt_res = float(state.best_opt_res)
    opt_tol = float(state.target_opt_tol)
    if isinstance(state, (ConstrainedTracker, ConstrainedTrackerMixin)):
      feas_res = float(state.best_feas_res)
      feas_tol = float(state.target_feas_tol)
    else:
      feas_res = 0.0
      feas_tol = 0.0
    counts = state.counts
    elapsed = time.perf_counter() - self._t0

    self.outer_iters.append(outer)
    self.inner_iters.append(inner)
    self.obj_vals.append(obj)
    self.opt_residuals.append(opt_res)
    self.feas_residuals.append(feas_res)
    self.opt_tols.append(opt_tol)
    self.feas_tols.append(feas_tol)
    self.smooth_evals.append(int(counts.smooth_evals))
    self.grad_evals.append(int(counts.grad_evals))
    self.prox_evals.append(int(counts.prox_evals))
    self.elapsed_times.append(elapsed)

    should_print = (
        not self._printed_header
        or len(self.outer_iters) % self.print_every == 0
    )
    if self.verbose and should_print:
      if not self._printed_header:
        print(
            f"{'iter':>6} | {'inner':>7} | {'obj':>12} | "
            f"{'opt_res (tol)':>21} | {'feas_res (tol)':>21} | {'time':>7}"
        )
        print("-" * 88)
        self._printed_header = True
      print(
          f"{outer:>6d} | {inner:>7d} | {obj:>12.4e} | "
          f"{opt_res:>9.2e} ({opt_tol:>9.2e}) | "
          f"{feas_res:>9.2e} ({feas_tol:>9.2e}) | {elapsed:>6.2f}s"
      )
    return False

  def __len__(self) -> int:
    """Returns the number of recorded solver steps."""
    return len(self.outer_iters)

  def to_dict(self) -> dict[str, list[int] | list[float]]:
    """Returns all recorded trajectories as a plain dictionary."""
    return {
        "outer_iters": list(self.outer_iters),
        "inner_iters": list(self.inner_iters),
        "obj_vals": list(self.obj_vals),
        "opt_residuals": list(self.opt_residuals),
        "feas_residuals": list(self.feas_residuals),
        "opt_tols": list(self.opt_tols),
        "feas_tols": list(self.feas_tols),
        "smooth_evals": list(self.smooth_evals),
        "grad_evals": list(self.grad_evals),
        "prox_evals": list(self.prox_evals),
        "elapsed_times": list(self.elapsed_times),
    }

  def plot(
      self,
      *,
      x_axis: typing.Literal["iter", "time", "evals"] = "iter",
      title: str | None = None,
      show_tol: bool = True,
      alpha: float = 0.75,
      style: str | None = "mocha",
      ax: typing.Any = None,
  ) -> tuple[typing.Any, list[typing.Any]]:
    """Plots recorded objective and residual trajectories via Matplotlib."""
    from proxax import plotting  # noqa: PLC0415

    return plotting.plot_history(
        self,
        x_axis=x_axis,
        title=title,
        show_tol=show_tol,
        alpha=alpha,
        style=style,
        ax=ax,
    )

  def __repr__(self) -> str:
    """Returns a concise summary of recorded steps and latest metrics."""
    if not self.outer_iters:
      return "HistoryTracker(steps=0)"
    return (
        f"HistoryTracker(steps={len(self)}, "
        f"last_obj={self.obj_vals[-1]:.6e}, "
        f"opt_res={self.opt_residuals[-1]:.2e}/{self.opt_tols[-1]:.2e}, "
        f"feas_res={self.feas_residuals[-1]:.2e}/{self.feas_tols[-1]:.2e})"
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
