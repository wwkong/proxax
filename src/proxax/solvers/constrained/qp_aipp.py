"""Quadratic Penalty Accelerated Inexact Proximal Point (QP-AIPP) solver."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from proxax import _typing
from proxax import certificates
from proxax import problem as problem_mod
from proxax.oracle import composite
from proxax.solvers import base
from proxax.solvers.constrained import _penalty_common
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
    base_oracle, x_proj, dtype, bound_c = (
        _penalty_common.validate_constrained_problem(
            problem, x0, "QPAIPP", jit=jit, time_limit=time_limit
        )
    )

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

    c0, ref0, tracker0 = _penalty_common.init_penalty_tracker(
        base_oracle,
        bound_c,
        x_proj,
        dtype=dtype,
        penalty_init=self.penalty_init,
        lipschitz_init=L0_base,
        norm_kind=self.norm_kind,
        opt_tol=self.opt_tol,
        feas_tol=self.feas_tol,
        rel_opt_tol=self.rel_opt_tol,
        rel_feas_tol=self.rel_feas_tol,
        jit=jit,
        extra_smooth_evals=extra_s_evals,
        extra_grad_evals=extra_g_evals,
        extra_prox_evals=extra_p_evals,
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
    start_wall = _penalty_common.start_clock(jit=jit, time_limit=time_limit)

    def _body(st: QPAIPPState) -> QPAIPPState:
      pen_oracle = composite.PenalizedOracle(
          base=base_oracle,
          constraint=bound_c,
          multiplier=None,
          penalty=st.penalty,
          dampen=0.0,
      )
      rem_time = _penalty_common.remaining_time(jit, time_limit, start_wall)
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
      c_next, tracker = _penalty_common.advance_penalty_stage(
          st.tracker,
          pen_oracle,
          bound_c,
          base_oracle,
          cand_x=cand_x,
          cand_v=aipp_st.best_subgrad,
          cand_opt=aipp_st.best_opt_res,
          inner_iters=aipp_st.inner_iters,
          counts=aipp_st.counts,
          sub_status=aipp_st.status,
          penalty=st.penalty,
          penalty_factor=self.penalty_factor,
          penalty_max=self.penalty_max,
          max_penalty_iters=self.max_penalty_iters,
          norm_kind=self.norm_kind,
          dtype=dtype,
      )
      L_next = aipp_st.lipschitz * (c_next / st.penalty)

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
