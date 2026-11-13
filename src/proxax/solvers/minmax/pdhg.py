"""Primal-Dual Hybrid Gradient (`PDHG`) solver for min-max problems."""

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


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class PDHGState(base.SolverTrackerMixin):
  """Iteration state of the `PDHG` min-max solver."""

  x: _typing.PyTree
  y: _typing.PyTree
  x_bar: _typing.PyTree
  tau: jax.Array
  sigma: jax.Array
  tracker: base.SolverTracker


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class PDHG:
  """Primal-Dual Hybrid Gradient (Chambolle-Pock) solver for `MinMaxProblem`.

  Solves `min_x max_y Phi(x, y) + h(x) - g(y)` using over-relaxed primal-dual
  updates with adaptive step-size backtracking and exact primal-dual
  stationarity refinement.
  """

  opt_tol: float = 1e-5
  rel_opt_tol: float = 0.0
  tau_init: float = 0.25
  sigma_init: float = 0.25
  theta: float = 1.0
  max_iters: int = 2000
  eval_every: int = 10
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

    ref0 = minmax_cert.refine_minmax(
        oracle,
        x_init,
        y_init,
        tau=self.tau_init,
        sigma=self.sigma_init,
        norm_kind=self.norm_kind,
    )
    init_pair = (ref0.x_hat, ref0.y_hat)
    init_subgrad = (ref0.v_x, ref0.v_y)
    tracker0 = base.SolverTracker.from_init(
        init_pair,
        init_subgrad,
        ref0.residual,
        ref0.obj_val,
        opt_tol=self.opt_tol,
        rel_opt_tol=self.rel_opt_tol,
        opt_tol_override=opt_tol_override,
        smooth_evals=ref0.smooth_evals,
        grad_evals=ref0.grad_evals,
        prox_evals=ref0.prox_evals,
    )

    tau0 = jnp.asarray(self.tau_init, dtype=dtype)
    sigma0 = jnp.asarray(self.sigma_init, dtype=dtype)
    if problem.L_xx is not None and problem.L_xy is not None:
      lip_est = max(problem.L_xx + problem.L_xy, 1e-4)
      tau0 = jnp.minimum(tau0, jnp.asarray(0.5 / lip_est, dtype=dtype))
    if problem.L_yy is not None and problem.L_xy is not None:
      lip_est_y = max(problem.L_yy + problem.L_xy, 1e-4)
      sigma0 = jnp.minimum(sigma0, jnp.asarray(0.5 / lip_est_y, dtype=dtype))

    init_state = PDHGState(
        x=x_init,
        y=y_init,
        x_bar=x_init,
        tau=tau0,
        sigma=sigma0,
        tracker=tracker0,
    )

    def _body(st: PDHGState) -> PDHGState:
      gx = oracle.grad_x(st.x, st.y)
      x_next = oracle.prox_x.prox(
          tree.sub(st.x, tree.scale(st.tau, gx)),
          st.tau,
      )
      dx = tree.sub(x_next, st.x)
      x_bar_next = tree.add(x_next, tree.scale(self.theta, dx))
      gy_bar = oracle.grad_y(x_bar_next, st.y)
      y_next = oracle.prox_y.prox(
          tree.add(st.y, tree.scale(st.sigma, gy_bar)),
          st.sigma,
      )
      dy = tree.sub(y_next, st.y)

      # Adaptive step-size safeguard using local Lipschitz estimate.
      gx_next, gy_next = oracle.phi_val_grad(x_next, y_next)[1:]
      dgx = tree.sub(gx_next, gx)
      dgy = tree.sub(gy_next, gy_bar)
      step_norm_sq = tree.norm_sq(dx) + tree.norm_sq(dy)
      grad_diff_sq = tree.norm_sq(dgx) + tree.norm_sq(dgy)
      lip_sq = grad_diff_sq / jnp.maximum(
          step_norm_sq, jnp.asarray(1e-30, dtype=dtype)
      )
      lip_est = jnp.sqrt(lip_sq)
      step_max = jnp.maximum(st.tau, st.sigma)
      too_large = step_max * lip_est > jnp.asarray(0.85, dtype=dtype)
      tau_next = jnp.where(
          too_large,
          jnp.maximum(0.5 * st.tau, jnp.asarray(1e-6, dtype=dtype)),
          jnp.minimum(1.02 * st.tau, jnp.asarray(1.0, dtype=dtype)),
      )
      sigma_next = jnp.where(
          too_large,
          jnp.maximum(0.5 * st.sigma, jnp.asarray(1e-6, dtype=dtype)),
          jnp.minimum(1.02 * st.sigma, jnp.asarray(1.0, dtype=dtype)),
      )

      next_iter = st.tracker.outer_iters + 1
      do_eval = (next_iter % self.eval_every) == 0

      def _eval_ref() -> tuple[
          _typing.PyTree, _typing.PyTree, jax.Array, jax.Array, jax.Array
      ]:
        ref = minmax_cert.refine_minmax(
            oracle,
            x_next,
            y_next,
            tau=tau_next,
            sigma=sigma_next,
            norm_kind=self.norm_kind,
        )
        return (
            (ref.x_hat, ref.y_hat),
            (ref.v_x, ref.v_y),
            ref.residual,
            ref.obj_val,
            jnp.asarray(2, dtype=jnp.int32),
        )

      def _skip_ref() -> tuple[
          _typing.PyTree, _typing.PyTree, jax.Array, jax.Array, jax.Array
      ]:
        return (
            st.tracker.best_x,
            st.tracker.best_subgrad,
            jnp.asarray(jnp.inf, dtype=dtype),
            st.tracker.best_obj,
            jnp.asarray(0, dtype=jnp.int32),
        )

      cand_x, cand_v, cand_res, cand_obj, ref_evals = jax.lax.cond(
          do_eval, _eval_ref, _skip_ref
      )
      cand_finite = jnp.where(
          do_eval,
          jnp.logical_and(
              tree.all_finite(cand_x),
              jnp.isfinite(cand_res),
          ),
          tree.all_finite((x_next, y_next)),
      )

      tracker = st.tracker.update_incumbent(
          cand_x,
          cand_v,
          cand_res,
          cand_obj,
      ).advance(
          inner_inc=1,
          smooth_inc=2 + ref_evals,
          grad_inc=3 + ref_evals,
          prox_inc=2 + ref_evals,
          max_iters=self.max_iters,
          finite_ok=cand_finite,
      )
      return PDHGState(
          x=x_next,
          y=y_next,
          x_bar=x_bar_next,
          tau=tau_next,
          sigma=sigma_next,
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
