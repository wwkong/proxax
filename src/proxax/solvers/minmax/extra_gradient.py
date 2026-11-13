"""Proximal Extra-Gradient (Mirror-Prox) solver for min-max problems."""

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
class ExtraGradientState(base.SolverTrackerMixin):
  """Iteration state of the `ExtraGradient` min-max solver."""

  x: _typing.PyTree
  y: _typing.PyTree
  step_size: jax.Array
  tracker: base.SolverTracker


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ExtraGradient:
  """Adaptive Proximal Extra-Gradient solver for `MinMaxProblem`.

  Solves `min_x max_y Phi(x, y) + h(x) - g(y)` via Korpelevich / Mirror-Prox
  updates with backtracking step-size adaptation and zero-overhead exact
  primal-dual stationarity certification at the extrapolated midpoint.
  """

  opt_tol: float = 1e-5
  rel_opt_tol: float = 0.0
  step_init: float = 0.25
  step_min: float = 1e-6
  step_max: float = 2.0
  max_iters: int = 1000
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

    alpha0 = jnp.asarray(self.step_init, dtype=dtype)
    if (
        problem.L_xx is not None
        or problem.L_xy is not None
        or problem.L_yy is not None
    ):
      lip_sum = (
          (problem.L_xx or 0.0)
          + (problem.L_xy or 0.0)
          + (problem.L_yy or 0.0)
      )
      if lip_sum > 0.0:
        alpha0 = jnp.minimum(
            alpha0,
            jnp.asarray(0.5 / lip_sum, dtype=dtype),
        )

    ref0 = minmax_cert.refine_minmax(
        oracle,
        x_init,
        y_init,
        tau=alpha0,
        sigma=alpha0,
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

    init_state = ExtraGradientState(
        x=x_init,
        y=y_init,
        step_size=alpha0,
        tracker=tracker0,
    )

    def _body(st: ExtraGradientState) -> ExtraGradientState:
      _, gx, gy = oracle.phi_val_grad(st.x, st.y)

      def _ls_cond(
          carry: tuple[
              jax.Array,
              _typing.PyTree,
              _typing.PyTree,
              jax.Array,
              _typing.PyTree,
              _typing.PyTree,
              jax.Array,
              jax.Array,
          ],
      ) -> jax.Array:
        _, _, _, _, _, _, accepted, trials = carry
        return jnp.logical_and(jnp.logical_not(accepted), trials < 12)

      def _ls_body(
          carry: tuple[
              jax.Array,
              _typing.PyTree,
              _typing.PyTree,
              jax.Array,
              _typing.PyTree,
              _typing.PyTree,
              jax.Array,
              jax.Array,
          ],
      ) -> tuple[
          jax.Array,
          _typing.PyTree,
          _typing.PyTree,
          jax.Array,
          _typing.PyTree,
          _typing.PyTree,
          jax.Array,
          jax.Array,
      ]:
        alpha, _, _, _, _, _, _, trials = carry
        x_m = oracle.prox_x.prox(tree.sub(st.x, tree.scale(alpha, gx)), alpha)
        y_m = oracle.prox_y.prox(tree.add(st.y, tree.scale(alpha, gy)), alpha)
        val_m, gx_m, gy_m = oracle.phi_val_grad(x_m, y_m)
        dx_m = tree.sub(x_m, st.x)
        dy_m = tree.sub(y_m, st.y)
        dist_sq = tree.norm_sq(dx_m) + tree.norm_sq(dy_m)
        dgx = tree.sub(gx_m, gx)
        dgy = tree.sub(gy_m, gy)
        grad_diff_sq = tree.norm_sq(dgx) + tree.norm_sq(dgy)
        ok = jnp.logical_or(
            (alpha**2) * grad_diff_sq
            <= jnp.asarray(0.81, dtype=dtype) * dist_sq,
            alpha <= jnp.asarray(self.step_min, dtype=dtype),
        )
        alpha_next = jnp.where(
            ok,
            alpha,
            jnp.maximum(0.5 * alpha, jnp.asarray(self.step_min, dtype=dtype)),
        )
        return alpha_next, x_m, y_m, val_m, gx_m, gy_m, ok, trials + 1

      init_ls = (
          st.step_size,
          st.x,
          st.y,
          jnp.asarray(0.0, dtype=dtype),
          gx,
          gy,
          jnp.asarray(False),
          jnp.asarray(0, dtype=jnp.int32),
      )
      alpha_used, x_mid, y_mid, val_mid, gx_mid, gy_mid, _, trials = (
          jax.lax.while_loop(_ls_cond, _ls_body, init_ls)
      )

      # Exact primal-dual stationarity certificate at (x_mid, y_mid).
      v_x = tree.add(
          gx_mid,
          tree.sub(tree.scale(1.0 / alpha_used, tree.sub(st.x, x_mid)), gx),
      )
      v_y = tree.add(
          tree.scale(-1.0, gy_mid),
          tree.add(tree.scale(1.0 / alpha_used, tree.sub(st.y, y_mid)), gy),
      )
      res_mid = tree.norm(
          (v_x, v_y), kind=typing.cast(typing.Any, self.norm_kind)
      )
      obj_mid = (
          val_mid
          + oracle.prox_x.value(x_mid)
          - oracle.prox_y.value(y_mid)
      )

      # Full Extra-Gradient update.
      x_next = oracle.prox_x.prox(
          tree.sub(st.x, tree.scale(alpha_used, gx_mid)),
          alpha_used,
      )
      y_next = oracle.prox_y.prox(
          tree.add(st.y, tree.scale(alpha_used, gy_mid)),
          alpha_used,
      )

      alpha_next = jnp.minimum(
          jnp.where(trials == 1, 1.1 * alpha_used, alpha_used),
          jnp.asarray(self.step_max, dtype=dtype),
      )

      cand_finite = jnp.logical_and(
          tree.all_finite((x_mid, y_mid)),
          jnp.isfinite(res_mid),
      )
      tracker = st.tracker.update_incumbent(
          (x_mid, y_mid),
          (v_x, v_y),
          res_mid,
          obj_mid,
      ).advance(
          inner_inc=trials,
          smooth_inc=1 + trials,
          grad_inc=1 + trials,
          prox_inc=2 + 2 * trials,
          max_iters=self.max_iters,
          finite_ok=cand_finite,
      )
      return ExtraGradientState(
          x=x_next,
          y=y_next,
          step_size=alpha_next,
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
