"""Accelerated Inexact Proximal Point (AIPP) method for unconstrained NCO."""

from __future__ import annotations

import dataclasses
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
from proxax.solvers.inner import acg


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class AIPPState:
  """Outer iteration state of the `AIPP` method."""

  center: _typing.PyTree
  center_obj: jax.Array
  best_x: _typing.PyTree
  best_subgrad: _typing.PyTree
  best_opt_res: jax.Array
  best_obj: jax.Array
  target_opt_tol: jax.Array
  lam: jax.Array
  lipschitz: jax.Array
  outer_iters: jax.Array
  inner_iters: jax.Array
  smooth_evals: jax.Array
  grad_evals: jax.Array
  prox_evals: jax.Array
  status: jax.Array


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
    if opt_tol_override is not None:
      target_tol = jnp.asarray(opt_tol_override, dtype=dtype)
    else:
      target_tol = jnp.asarray(self.opt_tol, dtype=dtype) + (
          jnp.asarray(self.rel_opt_tol, dtype=dtype) * res0
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

    finite0 = jnp.logical_and(tree.all_finite(ref0.x_hat), jnp.isfinite(res0))
    status0 = jnp.where(
        jnp.logical_not(finite0),
        jnp.asarray(int(base.SolveStatus.NON_FINITE), dtype=jnp.int32),
        jnp.where(
            res0 <= target_tol,
            jnp.asarray(int(base.SolveStatus.CONVERGED), dtype=jnp.int32),
            jnp.asarray(int(base.SolveStatus.RUNNING), dtype=jnp.int32),
        ),
    )

    init_state = AIPPState(
        center=ref0.x_hat,
        center_obj=ref0.obj_val,
        best_x=ref0.x_hat,
        best_subgrad=ref0.v_hat,
        best_opt_res=res0,
        best_obj=ref0.obj_val,
        target_opt_tol=target_tol,
        lam=lam0,
        lipschitz=ref0.lipschitz_used,
        outer_iters=jnp.asarray(0, dtype=jnp.int32),
        inner_iters=jnp.asarray(0, dtype=jnp.int32),
        smooth_evals=ref0.smooth_evals,
        grad_evals=ref0.grad_evals,
        prox_evals=ref0.prox_evals,
        status=status0,
    )

    acg_cfg = acg.ACGConfig(
        sigma=self.sigma,
        max_iters=self.max_inner_iters,
        ls_decrease=self.inner_ls_decrease,
    )
    running_code = jnp.asarray(int(base.SolveStatus.RUNNING), dtype=jnp.int32)

    def _cond(st: AIPPState) -> jax.Array:
      return st.status == running_code

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
        next_outer = st.outer_iters + 1
        hit_max = next_outer >= self.max_outer_iters
        max_iter_code = jnp.asarray(
            int(base.SolveStatus.MAX_ITER_REACHED),
            dtype=jnp.int32,
        )
        next_status = jnp.where(
            hit_max,
            max_iter_code,
            running_code,
        )
        return AIPPState(
            center=st.center,
            center_obj=st.center_obj,
            best_x=st.best_x,
            best_subgrad=st.best_subgrad,
            best_opt_res=st.best_opt_res,
            best_obj=st.best_obj,
            target_opt_tol=st.target_opt_tol,
            lam=shrunk_lam,
            lipschitz=jnp.maximum(acg_st.lipschitz - smooth_reg, st.lipschitz),
            outer_iters=next_outer,
            inner_iters=st.inner_iters + acg_st.iteration,
            smooth_evals=st.smooth_evals + acg_st.smooth_evals + 1,
            grad_evals=st.grad_evals + acg_st.grad_evals,
            prox_evals=st.prox_evals + acg_st.prox_evals,
            status=next_status,
        )

      def _on_accept() -> AIPPState:
        est_base_L = jnp.maximum(
            acg_st.lipschitz - smooth_reg,
            0.5 * st.lipschitz,
        )
        ref = certificates.refine(oracle, acg_st.y, est_base_L, jit=jit)
        cand_res = tree.norm(ref.v_hat, kind=self.norm_kind)
        is_better = cand_res <= st.best_opt_res
        best_x = jax.tree.map(
            lambda cx, bx: jnp.where(is_better, cx, bx),
            ref.x_hat,
            st.best_x,
        )
        best_v = jax.tree.map(
            lambda cv, bv: jnp.where(is_better, cv, bv),
            ref.v_hat,
            st.best_subgrad,
        )
        best_res = jnp.where(is_better, cand_res, st.best_opt_res)
        best_obj = jnp.where(is_better, ref.obj_val, st.best_obj)

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

        next_outer = st.outer_iters + 1
        finite_ok = jnp.logical_and(
            tree.all_finite(best_x),
            jnp.isfinite(best_res),
        )
        ls_fail_code = jnp.asarray(
            int(acg.ACGStatus.LINE_SEARCH_FAILED),
            dtype=jnp.int32,
        )
        next_status = jnp.where(
            jnp.logical_not(finite_ok),
            jnp.asarray(int(base.SolveStatus.NON_FINITE), dtype=jnp.int32),
            jnp.where(
                acg_st.status == ls_fail_code,
                jnp.asarray(
                    int(base.SolveStatus.LINE_SEARCH_FAILED),
                    dtype=jnp.int32,
                ),
                jnp.where(
                    best_res <= st.target_opt_tol,
                    jnp.asarray(
                        int(base.SolveStatus.CONVERGED),
                        dtype=jnp.int32,
                    ),
                    jnp.where(
                        next_outer >= self.max_outer_iters,
                        jnp.asarray(
                            int(base.SolveStatus.MAX_ITER_REACHED),
                            dtype=jnp.int32,
                        ),
                        running_code,
                    ),
                ),
            ),
        )

        return AIPPState(
            center=next_center,
            center_obj=next_center_obj,
            best_x=best_x,
            best_subgrad=best_v,
            best_opt_res=best_res,
            best_obj=best_obj,
            target_opt_tol=st.target_opt_tol,
            lam=next_lam,
            lipschitz=ref.lipschitz_used,
            outer_iters=next_outer,
            inner_iters=st.inner_iters + acg_st.iteration,
            smooth_evals=(
                st.smooth_evals + acg_st.smooth_evals + ref.smooth_evals + 1
            ),
            grad_evals=st.grad_evals + acg_st.grad_evals + ref.grad_evals,
            prox_evals=st.prox_evals + acg_st.prox_evals + ref.prox_evals,
            status=next_status,
        )

      if jit:
        return typing.cast(
            AIPPState,
            jax.lax.cond(step_rejected, _on_reject, _on_accept),
        )
      return _on_reject() if bool(step_rejected) else _on_accept()

    def _on_timeout(st: AIPPState) -> AIPPState:
      return dataclasses.replace(
          st,
          status=jnp.asarray(
              int(base.SolveStatus.TIME_LIMIT_REACHED),
              dtype=jnp.int32,
          ),
      )

    return _loop.while_loop(
        _cond,
        _body,
        init_state,
        jit=jit,
        callback=callback,
        time_limit=time_limit,
        on_time_limit=_on_timeout,
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
    fin = self.solve_oracle(
        problem.oracle,
        x0,
        jit=jit,
        callback=callback,
        time_limit=time_limit,
    )
    dtype = fin.best_opt_res.dtype
    stats = base.SolveStats(
        outer_iters=fin.outer_iters,
        inner_iters=fin.inner_iters,
        smooth_evals=fin.smooth_evals,
        grad_evals=fin.grad_evals,
        prox_evals=fin.prox_evals,
        opt_residual=fin.best_opt_res,
        feas_residual=jnp.asarray(0.0, dtype=dtype),
        obj_val=fin.best_obj,
    )
    return base.Solution(
        x=fin.best_x,
        dual=None,
        subgrad=fin.best_subgrad,
        status=fin.status,
        stats=stats,
    )
