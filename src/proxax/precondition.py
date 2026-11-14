"""Modified Ruiz equilibration and diagonal preconditioning for Proxax."""

from __future__ import annotations

import dataclasses
import typing

import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsparse

import proxax.solve as solve_mod
from proxax import _typing
from proxax import problem as problem_mod
from proxax.constraints import linear_op
from proxax.constraints import sets
from proxax.oracle import terms
from proxax.prox import base as prox_base
from proxax.prox import indicators
from proxax.prox import norms
from proxax.solvers import base


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class _DiagonalL1:
  """Elementwise-weighted L1 norm `sum_i weight_i * |x_i|`."""

  weight: jax.Array

  @property
  def strong_convexity(self) -> float:
    """Strong convexity constant (`0.0`)."""
    return 0.0

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates `sum_i weight_i * |x_i|`."""
    arr = jnp.asarray(x)
    return jnp.sum(self.weight * jnp.abs(arr))

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> jax.Array:
    """Applies elementwise soft-thresholding with `step_size * weight`."""
    arr = jnp.asarray(x)
    thresh = jnp.asarray(step_size, dtype=arr.dtype) * self.weight
    return jnp.sign(arr) * jnp.maximum(jnp.abs(arr) - thresh, 0.0)

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Identity projection (`dom(L1)` is the full space)."""
    return x


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class PreconditionedProblem:
  """Scaled `CompositeProblem` with original-space tolerance & solution maps.

  Represents the change of variables `x = d_x * x_scaled`, dual
  `dual = (d_y * dual_scaled) / cost_scale`, and scaled objective
  `cost_scale * f(d_x * x_scaled)`.
  """

  problem: problem_mod.CompositeProblem
  x0: jax.Array
  d_x: jax.Array
  d_y: jax.Array | None
  cost_scale: float
  opt_tol_factor: float
  feas_tol_factor: float

  def scale_opt_tol(self, opt_tol: float) -> float:
    """Scales `opt_tol` so scaled convergence certifies original `opt_tol`."""
    return opt_tol * self.opt_tol_factor

  def scale_feas_tol(self, feas_tol: float) -> float:
    """Scales `feas_tol` so scaled convergence certifies original `feas_tol`."""
    return feas_tol * self.feas_tol_factor

  def unscale(self, sol: base.Solution) -> base.Solution:
    """Maps a `Solution` from scaled coordinates back to original space."""
    x_orig = self.d_x * jnp.asarray(sol.x)
    subgrad_orig = jnp.asarray(sol.subgrad) / (self.cost_scale * self.d_x)
    dual_orig: _typing.PyTree | None = sol.dual
    if sol.dual is not None:
      if self.d_y is not None:
        dual_orig = (self.d_y * jnp.asarray(sol.dual)) / self.cost_scale
      else:
        dual_orig = jax.tree.map(lambda p: p / self.cost_scale, sol.dual)
    obj_orig = sol.stats.obj_val / self.cost_scale
    opt_res_orig = jnp.linalg.norm(subgrad_orig)
    feas_res_orig = sol.stats.feas_residual / self.feas_tol_factor
    stats_orig = dataclasses.replace(
        sol.stats,
        obj_val=obj_orig,
        opt_residual=opt_res_orig,
        feas_residual=feas_res_orig,
    )
    return dataclasses.replace(
        sol,
        x=x_orig,
        dual=dual_orig,
        subgrad=subgrad_orig,
        stats=stats_orig,
    )


def _row_max_abs(
    mat: jax.Array | jsparse.BCOO, axis: int, length: int
) -> jax.Array:
  """Computes `max_abs` along `axis` for dense or `BCOO` 2D matrices."""
  if isinstance(mat, jsparse.BCOO):
    out = jnp.zeros((length,), dtype=mat.data.dtype)
    if mat.nse == 0:
      return out
    idx_axis = 0 if axis == 1 else 1
    return out.at[mat.indices[:, idx_axis]].max(jnp.abs(mat.data))
  return jnp.max(jnp.abs(mat), axis=axis)


def _scale_matrix(
    mat: jax.Array | jsparse.BCOO,
    row_scale: jax.Array,
    col_scale: jax.Array,
) -> jax.Array | jsparse.BCOO:
  """Computes `diag(row_scale) @ mat @ diag(col_scale)` without densifying."""
  if isinstance(mat, jsparse.BCOO):
    scaled_data = (
        mat.data
        * row_scale[mat.indices[:, 0]]
        * col_scale[mat.indices[:, 1]]
    )
    return jsparse.BCOO((scaled_data, mat.indices), shape=mat.shape)
  return (row_scale[:, None] * mat) * col_scale[None, :]


def _can_scale_prox_diagonally(prox_op: typing.Any) -> bool:
  return isinstance(
      prox_op,
      prox_base.Zero | indicators.NonNegative | indicators.Box | norms.L1,
  )


def _scale_prox(
    prox_op: typing.Any,
    d_x: jax.Array,
    cost_scale: float,
) -> typing.Any:
  if isinstance(prox_op, prox_base.Zero | indicators.NonNegative):
    return prox_op
  if isinstance(prox_op, indicators.Box):
    lower = jnp.asarray(prox_op.lower, dtype=d_x.dtype) / d_x
    upper = jnp.asarray(prox_op.upper, dtype=d_x.dtype) / d_x
    return indicators.Box(lower=lower, upper=upper)
  if isinstance(prox_op, norms.L1):
    w = jnp.asarray(prox_op.weight, dtype=d_x.dtype)
    return _DiagonalL1(weight=cost_scale * w * d_x)
  raise TypeError(
      f"Diagonal primal scaling is unsupported for {type(prox_op).__name__}."
  )


def _scale_target_set(
    target_set: sets.ConvexSet,
    d_y: jax.Array,
) -> sets.ConvexSet | None:
  if isinstance(target_set, sets.ZeroSet):
    return target_set
  if isinstance(target_set, sets.Singleton):
    rhs = jnp.asarray(target_set.rhs, dtype=d_y.dtype) * d_y
    return sets.Singleton(rhs=rhs)
  if isinstance(target_set, sets.UpperBound):
    rhs = jnp.asarray(target_set.rhs, dtype=d_y.dtype) * d_y
    return sets.UpperBound(rhs=rhs)
  if isinstance(target_set, sets.LowerBound):
    rhs = jnp.asarray(target_set.rhs, dtype=d_y.dtype) * d_y
    return sets.LowerBound(rhs=rhs)
  if isinstance(target_set, sets.BoxSet):
    lower = jnp.asarray(target_set.lower, dtype=d_y.dtype) * d_y
    upper = jnp.asarray(target_set.upper, dtype=d_y.dtype) * d_y
    return sets.BoxSet(lower=lower, upper=upper)
  return None


def ruiz_equilibrate(
    problem: problem_mod.CompositeProblem,
    x0: jax.Array,
    *,
    hessian: jax.Array | jsparse.BCOO | None = None,
    num_iters: int = 10,
    min_scale: float = 1e-4,
    max_scale: float = 1e4,
) -> PreconditionedProblem:
  """Applies Modified Ruiz Equilibration to a 1D `CompositeProblem`.

  Supports both dense `jax.Array` and sparse `jax.experimental.sparse.BCOO`
  constraint and Hessian matrices without densification.

  Args:
    problem: Composite problem `min f(x) + h(x) s.t. A x in C`.
    x0: Initial 1D primal iterate.
    hessian: Optional Hessian / quadratic matrix `P` (dense or `BCOO`) used to
      include objective curvature in column scaling.
    num_iters: Number of Ruiz equilibration sweeps.
    min_scale: Lower clamp on per-step diagonal scaling factors.
    max_scale: Upper clamp on per-step diagonal scaling factors.

  Returns:
    A :class:`PreconditionedProblem` holding the scaled problem, initial point,
    tolerance multipliers, and `.unscale(sol)` helper.
  """
  x0_arr = jnp.asarray(x0)
  if x0_arr.ndim != 1:
    raise ValueError("`ruiz_equilibrate` currently expects a 1D primal array.")
  n = x0_arr.shape[0]
  dtype = x0_arr.dtype

  a_mat: jax.Array | jsparse.BCOO | None = None
  set_c: linear_op.SetConstraint | None = None
  if isinstance(problem.constraints, linear_op.SetConstraint) and isinstance(
      problem.constraints.operator, linear_op.MatrixLinearOperator
  ):
    cand_set = problem.constraints.target_set
    if _scale_target_set(cand_set, jnp.ones((1,), dtype=dtype)) is not None:
      set_c = problem.constraints
      a_mat = problem.constraints.operator.matrix

  allow_dx = _can_scale_prox_diagonally(problem.proximal)
  p_cur = hessian
  a_cur = a_mat
  m = a_cur.shape[0] if a_cur is not None else 0

  d_x = jnp.ones((n,), dtype=dtype)
  d_y = jnp.ones((m,), dtype=dtype) if a_cur is not None else None
  _, grad_x0 = problem.oracle.smooth_value_and_grad(x0_arr)
  grad_cur = jnp.asarray(grad_x0, dtype=dtype)
  cost_scale = 1.0

  for _ in range(num_iters):
    if allow_dx:
      norm_p = (
          _row_max_abs(p_cur, axis=1, length=n)
          if p_cur is not None
          else jnp.zeros((n,), dtype=dtype)
      )
      norm_at = (
          _row_max_abs(a_cur, axis=0, length=n)
          if a_cur is not None and m > 0
          else jnp.zeros((n,), dtype=dtype)
      )
      rx = jnp.maximum(norm_p, norm_at)
      sx = jnp.where(rx > 1e-12, 1.0 / jnp.sqrt(rx), 1.0)
      sx = jnp.clip(sx, min_scale, max_scale)
      d_x = d_x * sx
      grad_cur = sx * grad_cur
      if p_cur is not None:
        p_cur = _scale_matrix(p_cur, sx, sx)
      if a_cur is not None:
        a_cur = _scale_matrix(a_cur, jnp.ones((m,), dtype=dtype), sx)

    if a_cur is not None and m > 0 and d_y is not None:
      ry = _row_max_abs(a_cur, axis=1, length=m)
      sy = jnp.where(ry > 1e-12, 1.0 / jnp.sqrt(ry), 1.0)
      sy = jnp.clip(sy, min_scale, max_scale)
      d_y = d_y * sy
      a_cur = _scale_matrix(a_cur, sy, jnp.ones((n,), dtype=dtype))

    mean_p = (
        float(jnp.mean(_row_max_abs(p_cur, axis=1, length=n)))
        if p_cur is not None
        else 0.0
    )
    norm_g = float(jnp.max(jnp.abs(grad_cur))) if n > 0 else 0.0
    denom = max(mean_p, norm_g, 1e-12)
    gamma = float(jnp.clip(1.0 / denom, min_scale, max_scale))
    if p_cur is not None:
      p_cur = (
          jsparse.BCOO((gamma * p_cur.data, p_cur.indices), shape=p_cur.shape)
          if isinstance(p_cur, jsparse.BCOO)
          else gamma * p_cur
      )
    grad_cur = gamma * grad_cur
    cost_scale *= gamma

  base_smooth = typing.cast(terms.SmoothObjective, problem.smooth)

  def _scaled_smooth(x_scaled: jax.Array) -> jax.Array:
    val = base_smooth.value(d_x * x_scaled)
    return jnp.asarray(cost_scale, dtype=val.dtype) * val

  scaled_prox = (
      _scale_prox(problem.proximal, d_x, cost_scale)
      if allow_dx
      else problem.proximal
  )

  scaled_constraints = problem.constraints
  if set_c is not None and a_cur is not None and d_y is not None:
    scaled_set = _scale_target_set(set_c.target_set, d_y)
    assert scaled_set is not None
    scaled_constraints = linear_op.SetConstraint(
        operator=linear_op.MatrixLinearOperator(matrix=a_cur),
        target_set=scaled_set,
    )

  scaled_problem = problem_mod.CompositeProblem(
      smooth=terms.as_smooth_term(_scaled_smooth),
      proximal=scaled_prox,
      constraints=scaled_constraints,
  )
  opt_factor = min(1.0, cost_scale * float(jnp.min(d_x))) if n > 0 else 1.0
  feas_factor = (
      min(1.0, float(jnp.min(d_y))) if d_y is not None and m > 0 else 1.0
  )
  return PreconditionedProblem(
      problem=scaled_problem,
      x0=x0_arr / d_x,
      d_x=d_x,
      d_y=d_y,
      cost_scale=cost_scale,
      opt_tol_factor=opt_factor,
      feas_tol_factor=feas_factor,
  )


def solve_preconditioned(
    problem: problem_mod.CompositeProblem,
    x0: jax.Array,
    *,
    solver: typing.Any = None,
    method: str | None = None,
    opt_tol: float = 1e-5,
    feas_tol: float = 1e-5,
    hessian: jax.Array | jsparse.BCOO | None = None,
    num_iters: int = 10,
    jit: bool = True,
    time_limit: float | None = None,
) -> base.Solution:
  """Equilibrates `problem` with Ruiz scaling, solves, and unscales."""
  pre = ruiz_equilibrate(problem, x0, hessian=hessian, num_iters=num_iters)
  eff_opt_tol = pre.scale_opt_tol(opt_tol)
  eff_feas_tol = pre.scale_feas_tol(feas_tol)
  if solver is not None:
    sol = solve_mod.solve(
        pre.problem,
        pre.x0,
        solver=solver,
        opt_tol=eff_opt_tol,
        feas_tol=eff_feas_tol,
        jit=jit,
        time_limit=time_limit,
    )
  else:
    sol = solve_mod.solve(
        pre.problem,
        pre.x0,
        method=method,
        opt_tol=eff_opt_tol,
        feas_tol=eff_feas_tol,
        jit=jit,
        time_limit=time_limit,
    )
  return pre.unscale(sol)


__all__ = [
    "PreconditionedProblem",
    "ruiz_equilibrate",
    "solve_preconditioned",
]
