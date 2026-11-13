"""Accelerated Inexact Proximal Point (AIPP) method for unconstrained NCO."""

from __future__ import annotations

import dataclasses
import typing

import jax
import jax.numpy as jnp

from proxax import _typing
from proxax import certificates
from proxax import problem as problem_mod
from proxax import tree
from proxax.oracle import composite
from proxax.solvers import base
from proxax.solvers.inner import acg


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class AIPPState(base.SolverTrackerMixin):
  """Outer iteration state of the `AIPP` method."""

  center: _typing.PyTree
  center_obj: jax.Array
  lam: jax.Array
  lipschitz: jax.Array
  tracker: base.SolverTracker


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class AIPP:
  """Accelerated Inexact Proximal Point solver (classical & adaptive modes).

  Solves `min_x f(x) + h(x)` to a certified stationary pair `(x_hat, v_hat)`
  with `v_hat in grad_f(x_hat) + dh(x_hat)` and `||v_hat|| <= opt_tol`.

  When `oracle.curvature_lower` (`m`) is provided and `adaptive_prox` is not
  forced True, runs classical fixed-`lam` `AIPP` with `lam = 0.5 / m` (or user
  `lam_init`). When `m` is unknown, automatically enables adaptive proximal step
  sizing with exact convexity and descent safeguarding (`R-AIPP` style).

  Attributes:
    opt_tol: Absolute stationarity tolerance `rho > 0`.
    rel_opt_tol: Optional relative stationarity tolerance against initial norm.
    norm_kind: Residual norm (`'l2'` or `'linf'`).
    sigma: Inner `ACG` relative inexactness tolerance in `(0, 1)`.
    lam_init: Optional initial proximal step size `lambda_0 > 0`.
    lam_min: Minimum allowed proximal step size during adaptation.
    lam_max: Maximum allowed proximal step size during adaptation.
    lam_increase: Multiplicative growth factor for `lam` after fast inner runs.
    lam_decrease: Multiplicative shrink factor (`< 1`) when nonconvexity hits.
    adaptive_prox: Whether to adapt `lam` dynamically (`None` = automatic).
    lipschitz_init: Initial smoothness estimate `L_0 > 0`.
    max_outer_iters: Maximum number of outer proximal point steps.
    max_inner_iters: Maximum `ACG` iterations per proximal subproblem.
    inner_ls_decrease: Local smoothness relaxation factor inside `ACG`.
  """

  opt_tol: float = 1e-5
  rel_opt_tol: float = 0.0
  norm_kind: str = "l2"
  sigma: float = 0.3
  lam_init: float | None = None
  lam_min: float = 1e-10
  lam_max: float = 1e8
  lam_increase: float = 1.25
  lam_decrease: float = 0.5
  adaptive_prox: bool | None = None
  lipschitz_init: float = 1.0
  max_outer_iters: int = 500
  max_inner_iters: int = 1000
  inner_ls_decrease: float = 0.9

  def solve_oracle(
      self,
      oracle: composite.CompositeOracleProtocol,
      x0: _typing.PyTree,
      *,
      lipschitz_init: _typing.Scalar | None = None,
      lam_init: _typing.Scalar | None = None,
      opt_tol_override: _typing.Scalar | None = None,
      jit: bool = True,
      callback: _typing.CallbackFn[AIPPState] | None = None,
      time_limit: float | None = None,
  ) -> AIPPState:
    """Runs `AIPP` directly on any `CompositeOracleProtocol`."""
    x_proj = oracle.domain_project(tree.as_float_tree(x0))
    dtype = tree.infer_dtype(x_proj)
    L_init = (
        jnp.asarray(self.lipschitz_init, dtype=dtype)
        if lipschitz_init is None
        else jnp.asarray(lipschitz_init, dtype=dtype)
    )

    # Initial refinement at projected x0 calibrates L_0 and checks step-0 stop.
    ref0 = certificates.refine(oracle, x_proj, L_init, jit=jit)
    res0 = tree.norm(ref0.v_hat, kind=self.norm_kind)
    tracker0 = base.SolverTracker.from_init(
        ref0.x_hat,
        ref0.v_hat,
        res0,
        ref0.obj_val,
        opt_tol=self.opt_tol,
        rel_opt_tol=self.rel_opt_tol,
        opt_tol_override=opt_tol_override,
        smooth_evals=ref0.smooth_evals,
        grad_evals=ref0.grad_evals,
        prox_evals=ref0.prox_evals,
    )

    has_known_m = oracle.curvature_lower is not None
    use_adaptive = (
        (not has_known_m) if self.adaptive_prox is None else self.adaptive_prox
    )

    if lam_init is not None:
      lam0 = jnp.asarray(lam_init, dtype=dtype)
    elif self.lam_init is not None:
      lam0 = jnp.asarray(self.lam_init, dtype=dtype)
    elif has_known_m:
      m_val = jnp.asarray(oracle.curvature_lower, dtype=dtype)
      lam0 = jnp.where(
          m_val > 0.0,
          0.5 / jnp.maximum(m_val, jnp.asarray(1e-12, dtype=dtype)),
          10.0
          / jnp.maximum(
              ref0.lipschitz_used,
              jnp.asarray(1e-6, dtype=dtype),
          ),
      )
    else:
      lam0 = jnp.clip(
          2.0
          / jnp.maximum(
              ref0.lipschitz_used,
              jnp.asarray(1e-6, dtype=dtype),
          ),
          jnp.asarray(self.lam_min, dtype=dtype),
          jnp.asarray(self.lam_max, dtype=dtype),
      )

    init_state = AIPPState(
        center=ref0.x_hat,
        center_obj=ref0.obj_val,
        lam=lam0,
        lipschitz=ref0.lipschitz_used,
        tracker=tracker0,
    )

    acg_cfg = acg.ACGConfig(
        sigma=self.sigma,
        max_iters=self.max_inner_iters,
        ls_decrease=self.inner_ls_decrease,
    )

    def _body(st: AIPPState) -> AIPPState:
      inv_lam = 1.0 / st.lam
      if has_known_m and not use_adaptive:
        m_arr = jnp.asarray(oracle.curvature_lower, dtype=dtype)
        smooth_reg = jnp.minimum(m_arr, inv_lam)
        prox_reg = jnp.maximum(jnp.asarray(0.0, dtype=dtype), inv_lam - m_arr)
      else:
        # Split 1/lam evenly so ACG enjoys strong convexity 0.5/lam while
        # testing whether local negative curvature <= 0.5/lam.
        smooth_reg = 0.5 * inv_lam
        prox_reg = 0.5 * inv_lam

      sub_oracle = composite.ProxRegularizedOracle(
          base=oracle,
          center=st.center,
          smooth_reg=smooth_reg,
          prox_reg=prox_reg,
      )
      sub_L_init = st.lipschitz + smooth_reg
      acg_st = acg.run_acg(
          sub_oracle,
          st.center,
          prox_center=st.center,
          lam=st.lam,
          lipschitz_init=sub_L_init,
          config=acg_cfg,
          check_convexity=use_adaptive,
          jit=jit,
      )

      # Evaluate original composite objective at y_t to verify descent.
      phi_y = oracle.value(acg_st.y)
      rhs_vec = tree.add(
          tree.scale(inv_lam, tree.sub(st.center, acg_st.y)),
          acg_st.u,
      )
      req_descent = (
          0.5 * st.lam * (1.0 - self.sigma**2) * tree.norm_sq(rhs_vec)
      )
      desc_slack = 100.0 * tree.eps(dtype) * (1.0 + jnp.abs(st.center_obj))
      descent_ok = (st.center_obj - phi_y + desc_slack) >= req_descent
      noncvx_code = jnp.asarray(
          int(acg.ACGStatus.NONCONVEX_DETECTED),
          dtype=jnp.int32,
      )
      step_rejected = jnp.logical_and(
          jnp.asarray(use_adaptive),
          jnp.logical_or(
              acg_st.status == noncvx_code,
              jnp.logical_not(descent_ok),
          ),
      )

      def _on_reject() -> AIPPState:
        shrunk_lam = jnp.maximum(
            st.lam * self.lam_decrease,
            jnp.asarray(self.lam_min, dtype=dtype),
        )
        next_tr = st.tracker.advance(
            inner_inc=acg_st.iteration,
            smooth_inc=1,
            counts_inc=acg_st.counts,
            max_iters=self.max_outer_iters,
            check_converged=False,
        )
        return AIPPState(
            center=st.center,
            center_obj=st.center_obj,
            lam=shrunk_lam,
            lipschitz=jnp.maximum(acg_st.lipschitz - smooth_reg, st.lipschitz),
            tracker=next_tr,
        )

      def _on_accept() -> AIPPState:
        est_base_L = jnp.maximum(
            acg_st.lipschitz - smooth_reg,
            0.5 * st.lipschitz,
        )
        ref = certificates.refine(oracle, acg_st.y, est_base_L, jit=jit)
        cand_res = tree.norm(ref.v_hat, kind=self.norm_kind)

        # Advance proximal center to whichever of y_t or x_hat has smaller obj.
        use_ref_center = ref.obj_val <= phi_y
        next_center = jax.tree.map(
            lambda rx, yx: jnp.where(use_ref_center, rx, yx),
            ref.x_hat,
            acg_st.y,
        )
        next_center_obj = jnp.where(use_ref_center, ref.obj_val, phi_y)

        grow_lam = jnp.logical_and(
            jnp.asarray(use_adaptive),
            acg_st.iteration <= 8,
        )
        next_lam = jnp.where(
            grow_lam,
            jnp.minimum(
                st.lam * self.lam_increase,
                jnp.asarray(self.lam_max, dtype=dtype),
            ),
            st.lam,
        )

        ls_fail_code = jnp.asarray(
            int(acg.ACGStatus.LINE_SEARCH_FAILED),
            dtype=jnp.int32,
        )
        next_tr = st.tracker.update_incumbent(
            ref.x_hat, ref.v_hat, cand_res, ref.obj_val
        ).advance(
            inner_inc=acg_st.iteration,
            smooth_inc=ref.smooth_evals + 1,
            grad_inc=ref.grad_evals,
            prox_inc=ref.prox_evals,
            counts_inc=acg_st.counts,
            max_iters=self.max_outer_iters,
            line_search_failed=acg_st.status == ls_fail_code,
        )

        return AIPPState(
            center=next_center,
            center_obj=next_center_obj,
            lam=next_lam,
            lipschitz=ref.lipschitz_used,
            tracker=next_tr,
        )

      if jit:
        return typing.cast(
            AIPPState,
            jax.lax.cond(step_rejected, _on_reject, _on_accept),
        )
      return _on_reject() if bool(step_rejected) else _on_accept()

    return base.run_solver_loop(
        _body,
        init_state,
        jit=jit,
        callback=callback,
        time_limit=time_limit,
    )

  def solve(
      self,
      problem: problem_mod.CompositeProblem,
      x0: _typing.PyTree,
      *,
      jit: bool = True,
      callback: _typing.CallbackFn[AIPPState] | None = None,
      time_limit: float | None = None,
  ) -> base.Solution:
    """Solves an unconstrained `CompositeProblem` with `AIPP`."""
    if problem.is_constrained:
      raise ValueError(
          "AIPP solves unconstrained composite problems; use QPAIPP for "
          "problems with `constraints`."
      )
    return self.solve_oracle(
        problem.oracle,
        x0,
        jit=jit,
        callback=callback,
        time_limit=time_limit,
    ).to_solution()
