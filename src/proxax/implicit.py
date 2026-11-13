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


def implicit_solve(
    problem_fn: collections.abc.Callable[
        [_typing.PyTree], problem_mod.CompositeProblem
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
  """Solves a parameterized `CompositeProblem` with implicit `custom_vjp`.

  Differentiates `x_hat(params)` (and optionally `dual_hat(params)`) with
  respect to `params` by solving the adjoint linear system of the optimality
  fixed-point / KKT conditions via matrix-free `gmres` / `cg` / `bicgstab`,
  without unrolling inner `jax.lax.while_loop` iterations.

  Args:
    problem_fn: Callable mapping `params` to a `CompositeProblem`.
    x0: Initial primal PyTree iterate.
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
    return_dual: If True and the problem is constrained, returns
      `(x_hat, dual_hat)`; otherwise returns `x_hat`.

  Returns:
    Optimal primal PyTree `x_hat` (or `(x_hat, dual_hat)` if `return_dual`).
  """
  probe_prob = problem_fn(params)
  is_constrained = probe_prob.is_constrained

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
    if return_dual and is_constrained:
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
    out = (sol.x, sol.dual) if (return_dual and is_constrained) else sol.x
    return out, (sol.x, sol.dual, theta)

  def _bwd(
      res: tuple[_typing.PyTree, _typing.PyTree, _typing.PyTree],
      g_out: _typing.PyTree,
  ) -> tuple[_typing.PyTree, _typing.PyTree]:
    x_hat, p_hat, theta = res
    tau = jnp.asarray(step_size, dtype=tree.infer_dtype(x_hat))

    if not is_constrained:

      def fixed_point_unc(
          x: _typing.PyTree, th: _typing.PyTree
      ) -> _typing.PyTree:
        prob = problem_fn(th)
        orc = prob.oracle
        _, grad = orc.smooth_value_and_grad(x)
        x_next = orc.prox(tree.sub(x, tree.scale(tau, grad)), tau)
        return tree.sub(x, x_next)

      _, vjp_x = jax.vjp(lambda x: fixed_point_unc(x, theta), x_hat)
      _, vjp_th = jax.vjp(lambda th: fixed_point_unc(x_hat, th), theta)

      def _matvec_unc(v: _typing.PyTree) -> _typing.PyTree:
        return tree.add(vjp_x(v)[0], tree.scale(damping, v))

      u = _solve_adjoint_linear(
          _matvec_unc,
          g_out,
          linear_solver=linear_solver,
          tol=linear_tol,
          maxiter=linear_maxiter,
      )
      d_theta = vjp_th(tree.scale(-1.0, u))[0]
      return tree.zeros_like(x0), d_theta

    z_hat = (x_hat, p_hat)

    def fixed_point_kkt(
        z: tuple[_typing.PyTree, _typing.PyTree], th: _typing.PyTree
    ) -> tuple[_typing.PyTree, _typing.PyTree]:
      x, p = z
      prob = problem_fn(th)
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

    _, vjp_z = jax.vjp(lambda z: fixed_point_kkt(z, theta), z_hat)
    _, vjp_th = jax.vjp(lambda th: fixed_point_kkt(z_hat, th), theta)

    def _matvec_kkt(v: _typing.PyTree) -> _typing.PyTree:
      return tree.add(vjp_z(v)[0], tree.scale(damping, v))

    rhs_z = g_out if return_dual else (g_out, tree.zeros_like(p_hat))
    u_z = _solve_adjoint_linear(
        _matvec_kkt,
        rhs_z,
        linear_solver=linear_solver,
        tol=linear_tol,
        maxiter=linear_maxiter,
    )
    d_theta = vjp_th(tree.scale(-1.0, u_z))[0]
    return tree.zeros_like(x0), d_theta

  _solve_fn.defvjp(_fwd, _bwd)
  return _solve_fn(x0, params)
