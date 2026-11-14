"""Implicit differentiation through `px.solve` via `jax.custom_vjp`."""

from __future__ import annotations

import collections.abc
import typing

import jax
import jax.numpy as jnp

from proxax import _typing
from proxax import problem as problem_mod
from proxax import solve as solve_mod
from proxax import tree
from proxax.oracle import block as block_mod
from proxax.oracle import minmax as minmax_mod
from proxax.oracle import terms

DifferentiableProblem = (
    problem_mod.CompositeProblem
    | minmax_mod.MinMaxProblem
    | block_mod.BlockProblem
)


def _solve_adjoint_linear(
    matvec: collections.abc.Callable[[_typing.PyTree], _typing.PyTree],
    rhs: _typing.PyTree,
    *,
    linear_solver: str,
    tol: float,
    maxiter: int,
) -> _typing.PyTree:
  """Solves `matvec(u) = rhs` using a matrix-free JAX iterative solver."""
  if linear_solver not in ("gmres", "cg", "bicgstab"):
    raise ValueError(
        f"Unknown `linear_solver` {linear_solver!r}; expected 'gmres', 'cg', "
        "or 'bicgstab'."
    )
  solver_fn: typing.Any = getattr(jax.scipy.sparse.linalg, linear_solver)
  sol, _ = solver_fn(matvec, rhs, tol=tol, maxiter=maxiter)
  return sol


def _implicit_vjp_solve(
    fixed_point_fn: collections.abc.Callable[
        [_typing.PyTree, _typing.PyTree], _typing.PyTree
    ],
    z_hat: _typing.PyTree,
    theta: _typing.PyTree,
    rhs: _typing.PyTree,
    *,
    linear_solver: str,
    linear_tol: float,
    linear_maxiter: int,
    damping: float,
) -> _typing.PyTree:
  """Computes `d_theta` via implicit differentiation of `fixed_point_fn`."""
  _, vjp_z = jax.vjp(lambda z: fixed_point_fn(z, theta), z_hat)
  _, vjp_th = jax.vjp(lambda th: fixed_point_fn(z_hat, th), theta)

  def _matvec(v: _typing.PyTree) -> _typing.PyTree:
    return tree.add(vjp_z(v)[0], tree.scale(damping, v))

  u = _solve_adjoint_linear(
      _matvec,
      rhs,
      linear_solver=linear_solver,
      tol=linear_tol,
      maxiter=linear_maxiter,
  )
  return vjp_th(tree.scale(-1.0, u))[0]


def implicit_vjp_from_solution(
    problem_fn: collections.abc.Callable[
        [_typing.PyTree], DifferentiableProblem
    ],
    x0: _typing.PyTree,
    x_hat: _typing.PyTree,
    p_hat: _typing.PyTree,
    theta: _typing.PyTree,
    g_out: _typing.PyTree,
    *,
    is_minmax: bool,
    is_block: bool,
    has_dual: bool,
    step_size: float = 1.0,
    linear_solver: str = "gmres",
    linear_tol: float = 1e-5,
    linear_maxiter: int = 100,
    damping: float = 1e-6,
    return_dual: bool = False,
) -> _typing.PyTree:
  """Computes implicit VJP `d_theta` given pre-solved `(x_hat, p_hat)`."""
  tau = jnp.asarray(step_size, dtype=tree.infer_dtype(x_hat))

  if is_minmax:

    def fixed_point_minmax(
        z: tuple[_typing.PyTree, _typing.PyTree], th: _typing.PyTree
    ) -> tuple[_typing.PyTree, _typing.PyTree]:
      x, y = z
      prob = typing.cast(minmax_mod.MinMaxProblem, problem_fn(th))
      _, gx, gy = prob.oracle.phi_val_grad(x, y)
      x_next = prob.prox_x.prox(tree.sub(x, tree.scale(tau, gx)), tau)
      y_next = prob.prox_y.prox(tree.add(y, tree.scale(tau, gy)), tau)
      return (tree.sub(x, x_next), tree.sub(y, y_next))

    return _implicit_vjp_solve(
        fixed_point_minmax,
        x_hat,
        theta,
        g_out,
        linear_solver=linear_solver,
        linear_tol=linear_tol,
        linear_maxiter=linear_maxiter,
        damping=damping,
    )

  if is_block:

    def fixed_point_block(
        z: tuple[_typing.PyTree, _typing.PyTree], th: _typing.PyTree
    ) -> tuple[_typing.PyTree, _typing.PyTree]:
      x_blocks, p = z
      prob = typing.cast(block_mod.BlockProblem, problem_fn(th))
      x_tuple = tuple(x_blocks)
      ops = prob.bind_operators(x_tuple)
      f_blocks = []
      for blk, op, x_i in zip(prob.blocks, ops, x_tuple, strict=True):
        smooth = typing.cast(terms.SmoothObjective, blk.smooth)
        _, grad_i = smooth.value_and_grad(x_i)
        adj_p = op.rmatvec(p)
        x_next = blk.proximal.prox(
            tree.sub(x_i, tree.scale(tau, tree.add(grad_i, adj_p))),
            tau,
        )
        f_blocks.append(tree.sub(x_i, x_next))
      f_p = prob.coupling_residual(x_tuple, bound_ops=ops)
      fx_out: _typing.PyTree = (
          tuple(f_blocks) if isinstance(x_blocks, tuple) else f_blocks
      )
      return (fx_out, f_p)

    rhs_b = g_out if return_dual else (g_out, tree.zeros_like(p_hat))
    return _implicit_vjp_solve(
        fixed_point_block,
        (x_hat, p_hat),
        theta,
        rhs_b,
        linear_solver=linear_solver,
        linear_tol=linear_tol,
        linear_maxiter=linear_maxiter,
        damping=damping,
    )

  if not has_dual:

    def fixed_point_unc(
        x: _typing.PyTree, th: _typing.PyTree
    ) -> _typing.PyTree:
      prob = typing.cast(problem_mod.CompositeProblem, problem_fn(th))
      orc = prob.oracle
      _, grad = orc.smooth_value_and_grad(x)
      x_next = orc.prox(tree.sub(x, tree.scale(tau, grad)), tau)
      return tree.sub(x, x_next)

    return _implicit_vjp_solve(
        fixed_point_unc,
        x_hat,
        theta,
        g_out,
        linear_solver=linear_solver,
        linear_tol=linear_tol,
        linear_maxiter=linear_maxiter,
        damping=damping,
    )

  def fixed_point_kkt(
      z: tuple[_typing.PyTree, _typing.PyTree], th: _typing.PyTree
  ) -> tuple[_typing.PyTree, _typing.PyTree]:
    x, p = z
    prob = typing.cast(problem_mod.CompositeProblem, problem_fn(th))
    orc = prob.oracle
    bound_c = prob.bind_constraints(x)
    assert bound_c is not None
    _, grad = orc.smooth_value_and_grad(x)
    adj_p = bound_c.adjoint(p)
    x_next = orc.prox(
        tree.sub(x, tree.scale(tau, tree.add(grad, adj_p))),
        tau,
    )
    f_x = tree.sub(x, x_next)
    _, p_next = bound_c.penalty_and_dual(
        x, multiplier=p, penalty=1.0, dampen=0.0
    )
    f_p = tree.sub(p_next, p)
    return (f_x, f_p)

  rhs_z = g_out if return_dual else (g_out, tree.zeros_like(p_hat))
  return _implicit_vjp_solve(
      fixed_point_kkt,
      (x_hat, p_hat),
      theta,
      rhs_z,
      linear_solver=linear_solver,
      linear_tol=linear_tol,
      linear_maxiter=linear_maxiter,
      damping=damping,
  )


def implicit_solve(
    problem_fn: collections.abc.Callable[
        [_typing.PyTree], DifferentiableProblem
    ],
    x0: _typing.PyTree,
    params: _typing.PyTree,
    *,
    solver: typing.Any = None,
    method: str | None = None,
    opt_tol: float = 1e-5,
    feas_tol: float = 1e-5,
    step_size: float = 1.0,
    linear_solver: str = "gmres",
    linear_tol: float = 1e-5,
    linear_maxiter: int = 100,
    damping: float = 1e-6,
    return_dual: bool = False,
) -> _typing.PyTree:
  """Solves a parameterized problem with implicit `custom_vjp`.

  Supports `CompositeProblem` (unconstrained and constrained),
  `MinMaxProblem`, and `BlockProblem`. Differentiates `x_hat(params)` (and
  optionally `dual_hat(params)`) with respect to `params` by solving the
  adjoint linear system of the optimality fixed-point / KKT conditions via
  matrix-free `gmres` / `cg` / `bicgstab`, without unrolling inner loops.

  Args:
    problem_fn: Callable mapping `params` to a `CompositeProblem`,
      `MinMaxProblem`, or `BlockProblem`.
    x0: Initial primal PyTree iterate (or `(x0, y0)` for `MinMaxProblem`, or
      block tuple for `BlockProblem`).
    params: Differentiable parameter PyTree `theta`.
    solver: Optional explicit Proxax solver instance.
    method: Optional solver name string.
    opt_tol: Stationarity tolerance used when `solver` is `None`.
    feas_tol: Feasibility tolerance used when `solver` is `None`.
    step_size: Proximal step scale `tau > 0` used in the fixed-point map.
    linear_solver: Matrix-free adjoint solver (`'gmres'`, `'cg'`, `'bicgstab'`).
    linear_tol: Relative tolerance for the adjoint linear solver.
    linear_maxiter: Maximum iterations for the adjoint linear solver.
    damping: Small Tikhonov regularization added to the adjoint operator.
    return_dual: If True and the problem has dual multipliers (constrained
      `CompositeProblem` or `BlockProblem`), returns `(x_hat, dual_hat)`.

  Returns:
    Optimal primal PyTree `x_hat` (or `(x_hat, dual_hat)` if `return_dual`).
  """
  probe_prob = problem_fn(params)
  is_minmax = isinstance(probe_prob, minmax_mod.MinMaxProblem)
  is_block = isinstance(probe_prob, block_mod.BlockProblem)
  has_dual = is_block or (
      isinstance(probe_prob, problem_mod.CompositeProblem)
      and probe_prob.is_constrained
  )

  @jax.custom_vjp
  def _solve_fn(
      x_init: _typing.PyTree,
      theta: _typing.PyTree,
  ) -> _typing.PyTree:
    prob = problem_fn(theta)
    sol = solve_mod.solve(
        prob,
        x_init,
        solver=solver,
        method=method,
        opt_tol=opt_tol,
        feas_tol=feas_tol,
        jit=True,
    )
    if return_dual and has_dual:
      return (sol.x, sol.dual)
    return sol.x

  def _fwd(
      x_init: _typing.PyTree,
      theta: _typing.PyTree,
  ) -> tuple[
      _typing.PyTree, tuple[_typing.PyTree, _typing.PyTree, _typing.PyTree]
  ]:
    prob = problem_fn(theta)
    sol = solve_mod.solve(
        prob,
        x_init,
        solver=solver,
        method=method,
        opt_tol=opt_tol,
        feas_tol=feas_tol,
        jit=True,
    )
    out = (sol.x, sol.dual) if (return_dual and has_dual) else sol.x
    return out, (sol.x, sol.dual, theta)

  def _bwd(
      res: tuple[_typing.PyTree, _typing.PyTree, _typing.PyTree],
      g_out: _typing.PyTree,
  ) -> tuple[_typing.PyTree, _typing.PyTree]:
    x_hat, p_hat, theta = res
    d_theta = implicit_vjp_from_solution(
        problem_fn,
        x0,
        x_hat,
        p_hat,
        theta,
        g_out,
        is_minmax=is_minmax,
        is_block=is_block,
        has_dual=has_dual,
        step_size=step_size,
        linear_solver=linear_solver,
        linear_tol=linear_tol,
        linear_maxiter=linear_maxiter,
        damping=damping,
        return_dual=return_dual,
    )
    return tree.zeros_like(x0), d_theta

  _solve_fn.defvjp(_fwd, _bwd)
  return _solve_fn(x0, params)
