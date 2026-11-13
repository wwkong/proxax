"""Quadratic Penalty Parameter-Free Accelerated Proximal Descent (QP-APD)."""

from __future__ import annotations

import dataclasses
import time

import jax
import jax.numpy as jnp

from proxax import _typing
from proxax import problem as problem_mod
from proxax.oracle import composite
from proxax.solvers import base
from proxax.solvers.constrained import _penalty_common
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
    base_oracle, x_proj, dtype, bound_c = (
        _penalty_common.validate_constrained_problem(
            problem, x0, "QPAPD", jit=jit, time_limit=time_limit
        )
    )
    c0, ref0, tracker0 = _penalty_common.init_penalty_tracker(
        base_oracle,
        bound_c,
        x_proj,
        dtype=dtype,
        penalty_init=self.penalty_init,
        lipschitz_init=self.lipschitz_init,
        norm_kind=self.norm_kind,
        opt_tol=self.opt_tol,
        feas_tol=self.feas_tol,
        rel_opt_tol=self.rel_opt_tol,
        rel_feas_tol=self.rel_feas_tol,
        jit=jit,
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
      rem_time = _penalty_common.remaining_time(jit, time_limit, start_wall)
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
      c_next, tracker = _penalty_common.advance_penalty_stage(
          st.tracker,
          pen_oracle,
          bound_c,
          base_oracle,
          cand_x=cand_x,
          cand_v=apd_st.best_subgrad,
          cand_opt=apd_st.best_opt_res,
          inner_iters=apd_st.inner_iters,
          counts=apd_st.counts,
          sub_status=apd_st.status,
          penalty=st.penalty,
          penalty_factor=self.penalty_factor,
          penalty_max=self.penalty_max,
          max_penalty_iters=self.max_penalty_iters,
          norm_kind=self.norm_kind,
          dtype=dtype,
      )
      M_next = apd_st.upper_curvature * (c_next / st.penalty)

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
