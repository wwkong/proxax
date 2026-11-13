"""Quadratic Penalty Parameter-Free Accelerated Proximal Descent (QP-APD)."""

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
from proxax.solvers.unconstrained import apd as apd_mod


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class QPAPDState(base.ConstrainedTrackerMixin):
  """Outer homotopy iteration state of the `QPAPD` solver."""

  primal: _typing.PyTree
  penalty: jax.Array
  lower_curvature: jax.Array
  upper_curvature: jax.Array
  tracker: base.ConstrainedTracker


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class QPAPD:
  """Quadratic Penalty + `APD` method for constrained nonconvex optimization.

  Implements the parameter-free quadratic penalty scheme of Section 5 in
  W. Kong, *"Complexity-Optimal and Parameter-Free First-Order Methods for
  Finding Stationary Points of Composite Optimization Problems,"* SIAM Journal
  on Optimization, 34(3):3005--3032, 2024.

  Solves set/cone-constrained nonconvex composite problems
  `min_x f(x) + h(x)  s.t.  A(x) in C` by applying `APD` to Moreau-envelope
  quadratic penalty subproblems `f(x) + h(x) + (c_k / 2) dist_C(A(x))^2` as
  `c_k` increases geometrically. Because `(c_k / 2) dist_C(A(x))^2` is convex,
  the lower-curvature estimate `m` tracks only the nonconvexity of `f` and
  never inflates with `c_k`.

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
    alpha: Multiplicative growth factor (`> 1`) for lower curvature `m_hat`.
    beta: Multiplicative growth factor (`> 1`) for inner smoothness `L`.
    theta: Proximal descent constant (`> 2`, default `4.0`).
    rho: Optional inner relative proximal error tolerance in `(0, 1)`.
    decay_curvature: Whether `APD` applies practical curvature decay Eq. (6.1).
    lower_curvature_init: Optional initial lower curvature `m_0 > 0` for `f`.
    lower_curvature_min: Floor `m_min > 0` guarding curvature decay in
      `float32`.
    lower_curvature_max: Maximum allowed lower curvature estimate.
    lipschitz_init: Initial smoothness seed `L_0 > 0` for step-0 calibration.
    max_apd_iters: Maximum `APD` subproblem calls per penalty stage.
    max_inner_iters: Maximum `PF.ACG` steps per subproblem call.
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
  alpha: float = 2.0
  beta: float = 2.0
  theta: float = 4.0
  rho: float | None = None
  decay_curvature: bool = True
  lower_curvature_init: float | None = None
  lower_curvature_min: float = 1e-6
  lower_curvature_max: float = 1e12
  lipschitz_init: float = 1.0
  max_apd_iters: int = 250
  max_inner_iters: int = 1000

  def solve_state(
      self,
      problem: problem_mod.CompositeProblem,
      x0: _typing.PyTree,
      *,
      jit: bool = True,
      callback: _typing.CallbackFn[QPAPDState] | None = None,
      time_limit: float | None = None,
  ) -> QPAPDState:
    """Runs `QPAPD` and returns the full `QPAPDState`."""
    if not problem.is_constrained:
      raise ValueError(
          "QPAPD requires a constrained `CompositeProblem` (`constraints` "
          "must not be None); use `APD` or `proxax.solve` instead."
      )
    if not jit and time_limit is not None and time_limit < 0:
      raise ValueError("`time_limit` must be non-negative.")

    base_oracle = problem.oracle
    x_proj = base_oracle.domain_project(tree.as_float_tree(x0))
    dtype = tree.infer_dtype(x_proj)
    bound_c = problem.bind_constraints(x_proj)
    assert bound_c is not None

    c0 = jnp.asarray(self.penalty_init, dtype=dtype)
    pen0 = composite.PenalizedOracle(
        base=base_oracle,
        constraint=bound_c,
        multiplier=None,
        penalty=c0,
        dampen=0.0,
    )
    L0_seed = jnp.asarray(self.lipschitz_init, dtype=dtype)
    ref0 = certificates.refine(pen0, x_proj, L0_seed, jit=jit)
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
        smooth_evals=ref0.smooth_evals + 1,
        grad_evals=ref0.grad_evals,
        prox_evals=ref0.prox_evals,
    )

    # Calibrate initial lower curvature `m0` from the unpenalized smooth term
    # `f` (never from `c_0 * ||A||^2`). For convex `f`, Section 5 of Kong (2024)
    # sets `m = opt_tol`.
    m_min_arr = jnp.asarray(self.lower_curvature_min, dtype=dtype)
    m_max_arr = jnp.asarray(self.lower_curvature_max, dtype=dtype)
    if self.lower_curvature_init is not None:
      m0 = jnp.clip(
          jnp.asarray(self.lower_curvature_init, dtype=dtype),
          m_min_arr,
          m_max_arr,
      )
    elif base_oracle.curvature_lower is not None:
      if base_oracle.curvature_lower > 0.0:
        m0 = jnp.clip(
            jnp.asarray(base_oracle.curvature_lower, dtype=dtype),
            m_min_arr,
            m_max_arr,
        )
      else:
        m0 = jnp.clip(
            jnp.maximum(tracker0.target_opt_tol, m_min_arr),
            m_min_arr,
            m_max_arr,
        )
    else:
      m0 = jnp.clip(jnp.asarray(1.0, dtype=dtype), m_min_arr, m_max_arr)

    init_state = QPAPDState(
        primal=ref0.x_hat,
        penalty=c0,
        lower_curvature=m0,
        upper_curvature=ref0.lipschitz_used,
        tracker=tracker0,
    )

    apd_solver = apd_mod.APD(
        opt_tol=self.opt_tol,
        rel_opt_tol=0.0,
        norm_kind=self.norm_kind,
        alpha=self.alpha,
        beta=self.beta,
        theta=self.theta,
        rho=self.rho,
        decay_curvature=self.decay_curvature,
        lower_curvature_min=self.lower_curvature_min,
        lower_curvature_max=self.lower_curvature_max,
        lipschitz_init=self.lipschitz_init,
        max_outer_iters=self.max_apd_iters,
        max_inner_iters=self.max_inner_iters,
    )
    start_wall = (
        time.perf_counter() if (not jit and time_limit is not None) else 0.0
    )

    def _body(st: QPAPDState) -> QPAPDState:
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

      apd_st = apd_solver.solve_oracle(
          pen_oracle,
          st.primal,
          lower_curvature_init=st.lower_curvature,
          upper_curvature_init=st.upper_curvature,
          opt_tol_override=st.target_opt_tol,
          jit=jit,
          time_limit=rem_time,
      )

      cand_x = apd_st.best_x
      cand_v = apd_st.best_subgrad
      cand_p = pen_oracle.effective_dual(cand_x)
      cand_opt = apd_st.best_opt_res
      cand_feas = tree.norm(bound_c.residual(cand_x), kind=self.norm_kind)
      cand_obj = base_oracle.value(cand_x)

      c_next = jnp.minimum(
          st.penalty * self.penalty_factor,
          jnp.asarray(self.penalty_max, dtype=dtype),
      )
      M_next = apd_st.upper_curvature * (c_next / st.penalty)

      tracker = st.tracker.update_incumbent(
          cand_x,
          cand_p,
          cand_v,
          cand_opt,
          cand_feas,
          cand_obj,
      ).advance(
          inner_inc=apd_st.inner_iters,
          smooth_inc=1,
          counts_inc=apd_st.counts,
          max_iters=self.max_penalty_iters,
          sub_status=apd_st.status,
      )

      return QPAPDState(
          primal=cand_x,
          penalty=c_next,
          lower_curvature=apd_st.lower_curvature,
          upper_curvature=M_next,
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
      callback: _typing.CallbackFn[QPAPDState] | None = None,
      time_limit: float | None = None,
  ) -> base.Solution:
    """Solves a constrained `CompositeProblem` using `QPAPD`."""
    fin = self.solve_state(
        problem,
        x0,
        jit=jit,
        callback=callback,
        time_limit=time_limit,
    )
    return fin.to_solution()
