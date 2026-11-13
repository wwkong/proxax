"""Inexact Proximal-Point solver for nonconvex / weakly-convex min-max."""

from __future__ import annotations

import dataclasses
import typing

import jax
import jax.numpy as jnp

from proxax import _typing
from proxax import tree
from proxax.certificates import minmax as minmax_cert
from proxax.oracle import minmax
from proxax.solvers import base
from proxax.solvers.minmax import extra_gradient


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ProxPointMinMaxState(base.SolverTrackerMixin):
  """Outer iteration state of the `ProxPointMinMax` solver."""

  x: _typing.PyTree
  y: _typing.PyTree
  rho_x: jax.Array
  rho_y: jax.Array
  tracker: base.SolverTracker


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ProxPointMinMax:
  """Inexact Proximal-Point Min-Max solver for `MinMaxProblem`.

  Solves nonconvex-concave and weakly-convex-weakly-concave problems
  `min_x max_y Phi(x, y) + h(x) - g(y)` by solving a sequence of
  regularized saddle-point subproblems
  `Phi(x, y) + (rho_x / 2) ||x - x_k||^2 - (rho_y / 2) ||y - y_k||^2`
  via `ExtraGradient` and certifying exact primal-dual stationarity on the
  original problem.
  """

  opt_tol: float = 1e-5
  rel_opt_tol: float = 0.0
  rho_x: float | None = None
  rho_y: float | None = None
  max_outer_iters: int = 200
  max_inner_iters: int = 200
  inner_tol_factor: float = 0.25
  step_init: float = 0.25
  norm_kind: str = "l2"

  def solve(
      self,
      problem: minmax.MinMaxProblem,
      x0: tuple[_typing.PyTree, _typing.PyTree],
      *,
      jit: bool = True,
      callback: typing.Any = None,
      time_limit: float | None = None,
      opt_tol_override: jax.Array | None = None,
  ) -> base.Solution:
    """Solves a `MinMaxProblem` from initial primal-dual pair `x0=(x, y)`."""
    oracle = problem.oracle
    x_init, y_init = oracle.domain_project(x0[0], x0[1])
    dtype = tree.infer_dtype(x_init)

    rho_x_val = (
        self.rho_x
        if self.rho_x is not None
        else max(2.0 * (problem.m_x or 0.5), 1e-2)
    )
    rho_y_val = (
        self.rho_y
        if self.rho_y is not None
        else (2.0 * problem.m_y if problem.m_y is not None else 0.0)
    )
    rho_x0 = jnp.asarray(rho_x_val, dtype=dtype)
    rho_y0 = jnp.asarray(rho_y_val, dtype=dtype)

    ref0 = minmax_cert.refine_minmax(
        oracle,
        x_init,
        y_init,
        tau=self.step_init,
        sigma=self.step_init,
        norm_kind=self.norm_kind,
    )
    tracker0 = base.SolverTracker.from_init(
        (ref0.x_hat, ref0.y_hat),
        (ref0.v_x, ref0.v_y),
        ref0.residual,
        ref0.obj_val,
        opt_tol=self.opt_tol,
        rel_opt_tol=self.rel_opt_tol,
        opt_tol_override=opt_tol_override,
        smooth_evals=ref0.smooth_evals,
        grad_evals=ref0.grad_evals,
        prox_evals=ref0.prox_evals,
    )

    init_state = ProxPointMinMaxState(
        x=x_init,
        y=y_init,
        rho_x=rho_x0,
        rho_y=rho_y0,
        tracker=tracker0,
    )

    inner_solver = extra_gradient.ExtraGradient(
        opt_tol=self.opt_tol * self.inner_tol_factor,
        step_init=self.step_init,
        max_iters=self.max_inner_iters,
        norm_kind=self.norm_kind,
    )

    def _body(st: ProxPointMinMaxState) -> ProxPointMinMaxState:
      x_anchor, y_anchor = st.x, st.y

      def sub_coupling(x: _typing.PyTree, y: _typing.PyTree) -> jax.Array:
        dx = tree.sub(x, x_anchor)
        dy = tree.sub(y, y_anchor)
        return (
            oracle.coupling(x, y)
            + 0.5 * st.rho_x * tree.norm_sq(dx)
            - 0.5 * st.rho_y * tree.norm_sq(dy)
        )

      sub_prob = minmax.MinMaxProblem(
          coupling=sub_coupling,
          prox_x=problem.prox_x,
          prox_y=problem.prox_y,
      )
      sub_tol = jnp.maximum(
          jnp.asarray(self.opt_tol * self.inner_tol_factor, dtype=dtype),
          jnp.asarray(1e-8, dtype=dtype),
      )
      sub_sol = inner_solver.solve(
          sub_prob,
          (st.x, st.y),
          jit=jit,
          opt_tol_override=sub_tol,
      )
      x_next, y_next = sub_sol.x

      ref = minmax_cert.refine_minmax(
          oracle,
          x_next,
          y_next,
          tau=self.step_init,
          sigma=self.step_init,
          norm_kind=self.norm_kind,
      )
      cand_x = (ref.x_hat, ref.y_hat)
      cand_v = (ref.v_x, ref.v_y)
      cand_finite = jnp.logical_and(
          tree.all_finite(cand_x),
          jnp.isfinite(ref.residual),
      )

      tracker = st.tracker.update_incumbent(
          cand_x,
          cand_v,
          ref.residual,
          ref.obj_val,
      ).advance(
          inner_inc=sub_sol.stats.outer_iters,
          smooth_inc=sub_sol.stats.smooth_evals + ref.smooth_evals,
          grad_inc=sub_sol.stats.grad_evals + ref.grad_evals,
          prox_inc=sub_sol.stats.prox_evals + ref.prox_evals,
          max_iters=self.max_outer_iters,
          finite_ok=cand_finite,
      )
      return ProxPointMinMaxState(
          x=x_next,
          y=y_next,
          rho_x=st.rho_x,
          rho_y=st.rho_y,
          tracker=tracker,
      )

    fin = base.run_solver_loop(
        _body,
        init_state,
        jit=jit,
        callback=callback,
        time_limit=time_limit,
    )
    return typing.cast(base.Solution, fin.to_solution())
