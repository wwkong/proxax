"""Shared helpers for quadratic-penalty solvers (`QPAIPP`, `QPAPD`)."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp

from proxax import _loop
from proxax import _typing
from proxax import certificates
from proxax import problem as problem_mod
from proxax import tree
from proxax.oracle import composite
from proxax.solvers import base


def validate_constrained_problem(
    problem: problem_mod.CompositeProblem,
    x0: _typing.PyTree,
    solver_name: str,
    *,
    jit: bool,
    time_limit: float | None,
) -> tuple[typing.Any, _typing.PyTree, typing.Any, typing.Any]:
  """Validates constrained `problem` & `time_limit`, returning oracle & x0."""
  if not problem.is_constrained:
    raise ValueError(
        f"{solver_name} requires a constrained `CompositeProblem` "
        "(`constraints` must not be None); use `proxax.solve` instead."
    )
  if time_limit is not None and time_limit < 0:
    raise ValueError("`time_limit` must be non-negative.")

  base_oracle = problem.oracle
  x_proj = base_oracle.domain_project(tree.as_float_tree(x0))
  dtype = tree.infer_dtype(x_proj)
  bound_c = problem.bind_constraints(x_proj)
  assert bound_c is not None
  return base_oracle, x_proj, dtype, bound_c


def init_penalty_tracker(
    base_oracle: typing.Any,
    bound_c: typing.Any,
    x_proj: _typing.PyTree,
    *,
    dtype: typing.Any,
    penalty_init: float,
    lipschitz_init: float | jax.Array,
    norm_kind: str,
    opt_tol: float,
    feas_tol: float,
    rel_opt_tol: float,
    rel_feas_tol: float,
    jit: bool,
    extra_smooth_evals: jax.Array | int = 0,
    extra_grad_evals: jax.Array | int = 0,
    extra_prox_evals: jax.Array | int = 0,
) -> tuple[jax.Array, certificates.RefinementResult, base.ConstrainedTracker]:
  """Initializes `c0`, initial refinement, and `ConstrainedTracker`."""
  c0 = jnp.asarray(penalty_init, dtype=dtype)
  pen0 = composite.PenalizedOracle(
      base=base_oracle,
      constraint=bound_c,
      multiplier=None,
      penalty=c0,
      dampen=0.0,
  )
  l0_seed = jnp.asarray(lipschitz_init, dtype=dtype)
  ref0 = certificates.refine(pen0, x_proj, l0_seed, jit=jit)
  p0 = pen0.effective_dual(ref0.x_hat)
  opt_res0 = tree.norm(ref0.v_hat, kind=norm_kind)
  feas_res0 = tree.norm(bound_c.residual(ref0.x_hat), kind=norm_kind)
  obj0 = base_oracle.value(ref0.x_hat)

  tracker0 = base.ConstrainedTracker.from_init(
      ref0.x_hat,
      p0,
      ref0.v_hat,
      opt_res0,
      feas_res0,
      obj0,
      opt_tol=opt_tol,
      feas_tol=feas_tol,
      rel_opt_tol=rel_opt_tol,
      rel_feas_tol=rel_feas_tol,
      smooth_evals=extra_smooth_evals + ref0.smooth_evals + 1,
      grad_evals=extra_grad_evals + ref0.grad_evals,
      prox_evals=extra_prox_evals + ref0.prox_evals,
  )
  return c0, ref0, tracker0


def start_clock(*, jit: bool, time_limit: float | None) -> float | jax.Array:
  """Captures a wall-clock timestamp when `time_limit` is active."""
  if time_limit is None:
    return 0.0
  return _loop.start_clock(jit=jit)


def remaining_time(
    jit: bool,
    time_limit: float | jax.Array | None,
    start_wall: float | jax.Array,
) -> float | None:
  """Computes remaining wall-clock budget for an inner stage."""
  return _loop.remaining_time(jit, time_limit, start_wall)


def advance_penalty_stage(
    tracker: base.ConstrainedTracker,
    pen_oracle: composite.PenalizedOracle,
    bound_c: typing.Any,
    base_oracle: typing.Any,
    *,
    cand_x: _typing.PyTree,
    cand_v: _typing.PyTree,
    cand_opt: jax.Array,
    inner_iters: jax.Array,
    counts: base.OracleCounts,
    sub_status: jax.Array,
    penalty: jax.Array,
    penalty_factor: float,
    penalty_max: float,
    max_penalty_iters: int,
    norm_kind: str,
    dtype: typing.Any,
) -> tuple[jax.Array, base.ConstrainedTracker]:
  """Updates incumbent, penalty `c_next`, and `ConstrainedTracker`."""
  cand_p = pen_oracle.effective_dual(cand_x)
  cand_feas = tree.norm(bound_c.residual(cand_x), kind=norm_kind)
  cand_obj = base_oracle.value(cand_x)
  c_next = jnp.minimum(
      penalty * penalty_factor,
      jnp.asarray(penalty_max, dtype=dtype),
  )
  next_tracker = tracker.update_incumbent(
      cand_x,
      cand_p,
      cand_v,
      cand_opt,
      cand_feas,
      cand_obj,
  ).advance(
      inner_inc=inner_iters,
      smooth_inc=1,
      counts_inc=counts,
      max_iters=max_penalty_iters,
      sub_status=sub_status,
  )
  return c_next, next_tracker
