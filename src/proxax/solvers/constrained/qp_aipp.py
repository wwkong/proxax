"""Quadratic Penalty Accelerated Inexact Proximal Point (QP-AIPP) solver."""

from __future__ import annotations

import dataclasses
import time

import jax
import jax.numpy as jnp

from proxax import _typing
from proxax import certificates
from proxax import problem as problem_mod
from proxax import tree
from proxax.oracle import composite
from proxax.solvers import base
from proxax.solvers.unconstrained import aipp as aipp_mod


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class QPAIPPState(base.ConstrainedTrackerMixin):
  """Outer homotopy iteration state of the `QPAIPP` solver."""

  primal: _typing.PyTree
  penalty: jax.Array
  lam: jax.Array
  lipschitz: jax.Array
  tracker: base.ConstrainedTracker


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

    tracker0 = base.ConstrainedTracker.from_init(
        ref0.x_hat,
        p0,
        ref0.v_hat,
        opt_res0,
        feas_res0,
        obj0,
        opt_tol=self.opt_tol,
        feas_tol=self.feas_tol,
        rel_opt_tol=self.rel_opt_tol,
        rel_feas_tol=self.rel_feas_tol,
        smooth_evals=extra_s_evals + ref0.smooth_evals + 1,
        grad_evals=extra_g_evals + ref0.grad_evals,
        prox_evals=extra_p_evals + ref0.prox_evals,
    )

    init_state = QPAIPPState(
        primal=ref0.x_hat,
        penalty=c0,
        lam=lam0,
        lipschitz=ref0.lipschitz_used,
        tracker=tracker0,
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

      c_next = jnp.minimum(
          st.penalty * self.penalty_factor,
          jnp.asarray(self.penalty_max, dtype=dtype),
      )
      L_next = aipp_st.lipschitz * (c_next / st.penalty)

      tracker = st.tracker.update_incumbent(
          cand_x,
          cand_p,
          cand_v,
          cand_opt,
          cand_feas,
          cand_obj,
      ).advance(
          inner_inc=aipp_st.inner_iters,
          smooth_inc=1,
          counts_inc=aipp_st.counts,
          max_iters=self.max_penalty_iters,
          sub_status=aipp_st.status,
      )

      return QPAIPPState(
          primal=cand_x,
          penalty=c_next,
          lam=aipp_st.lam,
          lipschitz=L_next,
          tracker=tracker,
      )

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
    return fin.to_solution()
