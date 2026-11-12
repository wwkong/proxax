"""Quadratic Penalty Accelerated Inexact Proximal Point (QP-AIPP) solver."""

from __future__ import annotations

import dataclasses
import time

import jax
import jax.numpy as jnp

from proxax import _loop
from proxax import _typing
from proxax import certificates
from proxax import problem as problem_mod
from proxax import tree
from proxax.oracle import composite
from proxax.solvers import base
from proxax.solvers.unconstrained import aipp as aipp_mod


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class QPAIPPState:
  """Outer homotopy iteration state of the `QPAIPP` solver."""

  primal: _typing.PyTree
  best_x: _typing.PyTree
  best_dual: _typing.PyTree
  best_subgrad: _typing.PyTree
  best_opt_res: jax.Array
  best_feas_res: jax.Array
  best_kkt_score: jax.Array
  best_obj: jax.Array
  target_opt_tol: jax.Array
  target_feas_tol: jax.Array
  penalty: jax.Array
  lam: jax.Array
  lipschitz: jax.Array
  outer_iters: jax.Array
  inner_iters: jax.Array
  smooth_evals: jax.Array
  grad_evals: jax.Array
  prox_evals: jax.Array
  status: jax.Array


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class QPAIPP:
  """Quadratic Penalty + `AIPP` method for constrained nonconvex optimization.

  Solves set/cone-constrained nonconvex composite problems
  `min_x f(x) + h(x)  s.t.  A(x) in C` by applying `AIPP` to a sequence of
  Moreau-envelope quadratic penalty subproblems
  `f(x) + h(x) + (c_k / 2) dist_C(A(x))^2` as `c_k` increases geometrically.
  Returns a certified primal-dual-subgradient triple `(x_hat, p_hat, v_hat)`
  satisfying:

  - `v_hat in grad_f(x_hat) + dh(x_hat) + A^*(p_hat)` with
    `||v_hat|| <= opt_tol`,
  - `p_hat in N_C(Pi_C(A(x_hat)))`, and
  - `||A(x_hat) - Pi_C(A(x_hat))|| <= feas_tol`.

  Attributes:
    opt_tol: Absolute Lagrangian stationarity tolerance `rho > 0`.
    feas_tol: Absolute primal set-inclusion feasibility tolerance `eta > 0`.
    rel_opt_tol: Optional relative stationarity tolerance vs step-0 norm.
    rel_feas_tol: Optional relative feasibility tolerance vs step-0 norm.
    norm_kind: Residual norm (`'l2'` or `'linf'`).
    penalty_init: Initial quadratic penalty parameter `c_0 > 0`.
    penalty_factor: Multiplicative penalty growth factor (`> 1`).
    penalty_max: Maximum allowed penalty parameter `c_max`.
    max_penalty_iters: Maximum number of outer penalty homotopy stages.
    sigma: Inner `ACG` relative inexactness parameter in `(0, 1)`.
    lam_init: Optional initial proximal step size `lambda_0 > 0`.
    lam_min: Minimum allowed proximal step size during adaptation.
    lam_max: Maximum allowed proximal step size during adaptation.
    adaptive_prox: Whether `AIPP` adapts `lam` dynamically (`None` = auto).
    lipschitz_init: Initial smoothness estimate `L_0 > 0` for `f`.
    max_aipp_iters: Maximum `AIPP` proximal steps per penalty stage.
    max_inner_iters: Maximum `ACG` steps per proximal subproblem.
  """

  opt_tol: float = 1e-5
  feas_tol: float = 1e-5
  rel_opt_tol: float = 0.0
  rel_feas_tol: float = 0.0
  norm_kind: str = "l2"
  penalty_init: float = 1.0
  penalty_factor: float = 2.0
  penalty_max: float = 1e12
  max_penalty_iters: int = 35
  sigma: float = 0.3
  lam_init: float | None = None
  lam_min: float = 1e-10
  lam_max: float = 1e8
  adaptive_prox: bool | None = None
  lipschitz_init: float = 1.0
  max_aipp_iters: int = 250
  max_inner_iters: int = 1000

  def solve_state(
      self,
      problem: problem_mod.CompositeProblem,
      x0: _typing.PyTree,
      *,
      jit: bool = True,
      callback: _typing.CallbackFn[QPAIPPState] | None = None,
      time_limit: float | None = None,
  ) -> QPAIPPState:
    """Runs `QPAIPP` and returns the full `QPAIPPState`."""
    if not problem.is_constrained:
      raise ValueError(
          "QPAIPP requires a constrained `CompositeProblem` (`constraints` "
          "must not be None); use `AIPP` or `proxax.solve` instead."
      )
    if not jit and time_limit is not None and time_limit < 0:
      raise ValueError("`time_limit` must be non-negative.")

    base_oracle = problem.oracle
    x_proj = base_oracle.domain_project(tree.as_float_tree(x0))
    dtype = tree.infer_dtype(x_proj)
    bound_c = problem.bind_constraints(x_proj)
    assert bound_c is not None

    L0_base = jnp.asarray(self.lipschitz_init, dtype=dtype)
    has_known_m = base_oracle.curvature_lower is not None

    # Calibrate initial proximal step `lam0` from unpenalized curvature of `f`
    # so large penalty curvature `c * ||A||^2` (which is convex!) never shrinks
    # `lam` unnecessarily.
    extra_s_evals = jnp.asarray(0, dtype=jnp.int32)
    extra_g_evals = jnp.asarray(0, dtype=jnp.int32)
    extra_p_evals = jnp.asarray(0, dtype=jnp.int32)
    if self.lam_init is not None:
      lam0 = jnp.asarray(self.lam_init, dtype=dtype)
    elif has_known_m:
      m_val = jnp.asarray(base_oracle.curvature_lower, dtype=dtype)
      lam0 = jnp.where(
          m_val > 0.0,
          0.5 / jnp.maximum(m_val, jnp.asarray(1e-12, dtype=dtype)),
          10.0 / jnp.maximum(L0_base, jnp.asarray(1e-6, dtype=dtype)),
      )
    else:
      ref_base = certificates.refine(base_oracle, x_proj, L0_base, jit=jit)
      L0_base = ref_base.lipschitz_used
      lam0 = jnp.clip(
          2.0 / jnp.maximum(L0_base, jnp.asarray(1e-6, dtype=dtype)),
          jnp.asarray(self.lam_min, dtype=dtype),
          jnp.asarray(self.lam_max, dtype=dtype),
      )
      extra_s_evals = ref_base.smooth_evals
      extra_g_evals = ref_base.grad_evals
      extra_p_evals = ref_base.prox_evals

    c0 = jnp.asarray(self.penalty_init, dtype=dtype)
    pen0 = composite.PenalizedOracle(
        base=base_oracle,
        constraint=bound_c,
        multiplier=None,
        penalty=c0,
        dampen=0.0,
    )
    ref0 = certificates.refine(pen0, x_proj, L0_base, jit=jit)
    p0 = pen0.effective_dual(ref0.x_hat)
    opt_res0 = tree.norm(ref0.v_hat, kind=self.norm_kind)
    feas_res0 = tree.norm(bound_c.residual(ref0.x_hat), kind=self.norm_kind)
    obj0 = base_oracle.value(ref0.x_hat)

    target_opt = jnp.asarray(self.opt_tol, dtype=dtype) + (
        jnp.asarray(self.rel_opt_tol, dtype=dtype) * opt_res0
    )
    target_feas = jnp.asarray(self.feas_tol, dtype=dtype) + (
        jnp.asarray(self.rel_feas_tol, dtype=dtype) * feas_res0
    )
    tiny = jnp.asarray(1e-30, dtype=dtype)
    kkt_score0 = jnp.maximum(
        opt_res0 / jnp.maximum(target_opt, tiny),
        feas_res0 / jnp.maximum(target_feas, tiny),
    )

    finite0 = jnp.logical_and(
        tree.all_finite(ref0.x_hat),
        jnp.logical_and(jnp.isfinite(opt_res0), jnp.isfinite(feas_res0)),
    )
    converged0 = jnp.logical_and(
        opt_res0 <= target_opt,
        feas_res0 <= target_feas,
    )
    running_code = jnp.asarray(int(base.SolveStatus.RUNNING), dtype=jnp.int32)
    status0 = jnp.where(
        jnp.logical_not(finite0),
        jnp.asarray(int(base.SolveStatus.NON_FINITE), dtype=jnp.int32),
        jnp.where(
            converged0,
            jnp.asarray(int(base.SolveStatus.CONVERGED), dtype=jnp.int32),
            running_code,
        ),
    )

    init_state = QPAIPPState(
        primal=ref0.x_hat,
        best_x=ref0.x_hat,
        best_dual=p0,
        best_subgrad=ref0.v_hat,
        best_opt_res=opt_res0,
        best_feas_res=feas_res0,
        best_kkt_score=kkt_score0,
        best_obj=obj0,
        target_opt_tol=target_opt,
        target_feas_tol=target_feas,
        penalty=c0,
        lam=lam0,
        lipschitz=ref0.lipschitz_used,
        outer_iters=jnp.asarray(0, dtype=jnp.int32),
        inner_iters=jnp.asarray(0, dtype=jnp.int32),
        smooth_evals=extra_s_evals + ref0.smooth_evals + 1,
        grad_evals=extra_g_evals + ref0.grad_evals,
        prox_evals=extra_p_evals + ref0.prox_evals,
        status=status0,
    )

    aipp_solver = aipp_mod.AIPP(
        opt_tol=self.opt_tol,
        rel_opt_tol=0.0,
        norm_kind=self.norm_kind,
        sigma=self.sigma,
        lam_min=self.lam_min,
        lam_max=self.lam_max,
        adaptive_prox=self.adaptive_prox,
        lipschitz_init=self.lipschitz_init,
        max_outer_iters=self.max_aipp_iters,
        max_inner_iters=self.max_inner_iters,
    )
    start_wall = (
        time.perf_counter() if (not jit and time_limit is not None) else 0.0
    )

    def _cond(st: QPAIPPState) -> jax.Array:
      return st.status == running_code

    def _body(st: QPAIPPState) -> QPAIPPState:
      pen_oracle = composite.PenalizedOracle(
          base=base_oracle,
          constraint=bound_c,
          multiplier=None,
          penalty=st.penalty,
          dampen=0.0,
      )
      rem_time: float | None = None
      if not jit and time_limit is not None:
        rem_time = max(0.0, time_limit - (time.perf_counter() - start_wall))

      aipp_st = aipp_solver.solve_oracle(
          pen_oracle,
          st.primal,
          lipschitz_init=st.lipschitz,
          lam_init=st.lam,
          opt_tol_override=st.target_opt_tol,
          jit=jit,
          time_limit=rem_time,
      )

      cand_x = aipp_st.best_x
      cand_v = aipp_st.best_subgrad
      cand_p = pen_oracle.effective_dual(cand_x)
      cand_opt = aipp_st.best_opt_res
      cand_feas = tree.norm(bound_c.residual(cand_x), kind=self.norm_kind)
      cand_obj = base_oracle.value(cand_x)
      cand_score = jnp.maximum(
          cand_opt / jnp.maximum(st.target_opt_tol, tiny),
          cand_feas / jnp.maximum(st.target_feas_tol, tiny),
      )

      is_better = cand_score <= st.best_kkt_score
      best_x = jax.tree.map(
          lambda cx, bx: jnp.where(is_better, cx, bx),
          cand_x,
          st.best_x,
      )
      best_p = jax.tree.map(
          lambda cp, bp: jnp.where(is_better, cp, bp),
          cand_p,
          st.best_dual,
      )
      best_v = jax.tree.map(
          lambda cv, bv: jnp.where(is_better, cv, bv),
          cand_v,
          st.best_subgrad,
      )
      best_opt = jnp.where(is_better, cand_opt, st.best_opt_res)
      best_feas = jnp.where(is_better, cand_feas, st.best_feas_res)
      best_score = jnp.where(is_better, cand_score, st.best_kkt_score)
      best_obj = jnp.where(is_better, cand_obj, st.best_obj)

      next_outer = st.outer_iters + 1
      c_next = jnp.minimum(
          st.penalty * self.penalty_factor,
          jnp.asarray(self.penalty_max, dtype=dtype),
      )
      L_next = aipp_st.lipschitz * (c_next / st.penalty)

      finite_ok = jnp.logical_and(
          tree.all_finite(best_x),
          jnp.logical_and(jnp.isfinite(best_opt), jnp.isfinite(best_feas)),
      )
      kkt_converged = jnp.logical_and(
          best_opt <= st.target_opt_tol,
          best_feas <= st.target_feas_tol,
      )
      timeout_code = jnp.asarray(
          int(base.SolveStatus.TIME_LIMIT_REACHED),
          dtype=jnp.int32,
      )
      ls_fail_code = jnp.asarray(
          int(base.SolveStatus.LINE_SEARCH_FAILED),
          dtype=jnp.int32,
      )
      next_status = jnp.where(
          jnp.logical_not(finite_ok),
          jnp.asarray(int(base.SolveStatus.NON_FINITE), dtype=jnp.int32),
          jnp.where(
              aipp_st.status == timeout_code,
              timeout_code,
              jnp.where(
                  aipp_st.status == ls_fail_code,
                  ls_fail_code,
                  jnp.where(
                      kkt_converged,
                      jnp.asarray(
                          int(base.SolveStatus.CONVERGED),
                          dtype=jnp.int32,
                      ),
                      jnp.where(
                          next_outer >= self.max_penalty_iters,
                          jnp.asarray(
                              int(base.SolveStatus.MAX_ITER_REACHED),
                              dtype=jnp.int32,
                          ),
                          running_code,
                      ),
                  ),
              ),
          ),
      )

      return QPAIPPState(
          primal=cand_x,
          best_x=best_x,
          best_dual=best_p,
          best_subgrad=best_v,
          best_opt_res=best_opt,
          best_feas_res=best_feas,
          best_kkt_score=best_score,
          best_obj=best_obj,
          target_opt_tol=st.target_opt_tol,
          target_feas_tol=st.target_feas_tol,
          penalty=c_next,
          lam=aipp_st.lam,
          lipschitz=L_next,
          outer_iters=next_outer,
          inner_iters=st.inner_iters + aipp_st.inner_iters,
          smooth_evals=st.smooth_evals + aipp_st.smooth_evals + 1,
          grad_evals=st.grad_evals + aipp_st.grad_evals,
          prox_evals=st.prox_evals + aipp_st.prox_evals,
          status=next_status,
      )

    def _on_timeout(st: QPAIPPState) -> QPAIPPState:
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
      callback: _typing.CallbackFn[QPAIPPState] | None = None,
      time_limit: float | None = None,
  ) -> base.Solution:
    """Solves a constrained `CompositeProblem` using `QPAIPP`."""
    fin = self.solve_state(
        problem,
        x0,
        jit=jit,
        callback=callback,
        time_limit=time_limit,
    )
    stats = base.SolveStats(
        outer_iters=fin.outer_iters,
        inner_iters=fin.inner_iters,
        smooth_evals=fin.smooth_evals,
        grad_evals=fin.grad_evals,
        prox_evals=fin.prox_evals,
        opt_residual=fin.best_opt_res,
        feas_residual=fin.best_feas_res,
        obj_val=fin.best_obj,
    )
    return base.Solution(
        x=fin.best_x,
        dual=fin.best_dual,
        subgrad=fin.best_subgrad,
        status=fin.status,
        stats=stats,
    )
