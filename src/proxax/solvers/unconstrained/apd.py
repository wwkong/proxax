"""Parameter-Free Accelerated Proximal Descent (`PF.APD` / `APD`) solver."""

from __future__ import annotations

import dataclasses
import math
import typing

import jax
import jax.numpy as jnp

from proxax import _typing
from proxax import certificates
from proxax import problem as problem_mod
from proxax import tree
from proxax.oracle import composite
from proxax.solvers import base
from proxax.solvers.inner import pf_acg


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class APDState(base.SolverTrackerMixin):
  """Outer iteration state of Kong's `PF.APD` (`APD`) solver."""

  center: _typing.PyTree
  center_obj: jax.Array
  lower_curvature: jax.Array
  upper_curvature: jax.Array
  never_increased_m: jax.Array
  tracker: base.SolverTracker


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class APD:
  """Parameter-Free Accelerated Proximal Descent (`PF.APD` / `APD`) solver.

  Implements Algorithms 3.1--3.4 and the practical curvature decay modification
  Eq. (6.1) of W. Kong, *"Complexity-Optimal and Parameter-Free First-Order
  Methods for Finding Stationary Points of Composite Optimization Problems,"*
  SIAM Journal on Optimization, 34(3):3005--3032, 2024.

  Solves `min_x f(x) + h(x)` without requiring knowledge of weak convexity `m`,
  smoothness `M`, or boundedness of `dom(h)`. Every inner `PF.ACG` iterate
  carries an exact subdifferential certificate
  `v = 2 * m_hat * (r + z_k - y) in grad_f(y) + dh(y)` with zero post-refinement
  overhead.

  Attributes:
    opt_tol: Absolute stationarity tolerance `rho_opt > 0`.
    rel_opt_tol: Optional relative stationarity tolerance against step-0 norm.
    norm_kind: Residual norm (`'l2'` or `'linf'`).
    alpha: Multiplicative growth factor (`> 1`) for lower curvature `m_hat`.
    beta: Multiplicative growth factor (`> 1`) for inner smoothness `L`.
    theta: Proximal descent constant (`> 2`, default `4.0`).
    rho: Optional inner relative proximal error tolerance in `(0, 1)`. Defaults
      to `1 / sqrt(alpha)` when `decay_curvature=True` (Eq. (6.1)) and
      `1 / sqrt(2 * alpha)` when `decay_curvature=False` (`rho^2 * alpha < 1`).
    decay_curvature: If True, enables practical curvature decay Eq. (6.1) on
      both `L_0` and `m_{k+1}`. If False, runs theoretical bitonic `PF.APD`
      (Algorithm 3.4).
    lower_curvature_init: Optional initial lower curvature `m_0 > 0`. When
      `None`, uses `oracle.curvature_lower` if positive, `max(opt_tol,
      lower_curvature_min)` if `oracle.curvature_lower == 0`, else `1.0`.
    lower_curvature_min: Floor `m_min > 0` guarding curvature decay in
      `float32`.
    lower_curvature_max: Maximum allowed lower curvature estimate.
    upper_curvature_init: Optional initial upper curvature `M_0 > 0`. When
      `None`, calibrated automatically at `x0`.
    lipschitz_init: Initial trial smoothness `L_0 > 0` for step-0 calibration.
    max_outer_iters: Maximum outer subproblem calls (including `m` retries).
    max_inner_iters: Maximum `PF.ACG` steps per subproblem call.
  """

  opt_tol: float = 1e-5
  rel_opt_tol: float = 0.0
  norm_kind: str = "l2"
  alpha: float = 2.0
  beta: float = 2.0
  theta: float = 4.0
  rho: float | None = None
  decay_curvature: bool = True
  lower_curvature_init: float | None = None
  lower_curvature_min: float = 1e-6
  lower_curvature_max: float = 1e12
  upper_curvature_init: float | None = None
  lipschitz_init: float = 1.0
  max_outer_iters: int = 500
  max_inner_iters: int = 1000

  def __post_init__(self) -> None:
    """Validates `APD` hyperparameters."""
    if self.alpha <= 1.0:
      raise ValueError("`alpha` must be strictly greater than 1.")
    if self.beta <= 1.0:
      raise ValueError("`beta` must be strictly greater than 1.")
    if self.theta <= 2.0:
      raise ValueError("`theta` must be strictly greater than 2.")
    if self.rho is not None and not (0.0 < self.rho < 1.0):
      raise ValueError("`rho` must lie in `(0, 1)`.")

  @property
  def _effective_rho(self) -> float:
    if self.rho is not None:
      return self.rho
    if self.decay_curvature:
      return 1.0 / math.sqrt(self.alpha)
    return 1.0 / math.sqrt(2.0 * self.alpha)

  def solve_oracle(
      self,
      oracle: composite.CompositeOracleProtocol,
      x0: _typing.PyTree,
      *,
      lower_curvature_init: _typing.Scalar | None = None,
      upper_curvature_init: _typing.Scalar | None = None,
      opt_tol_override: _typing.Scalar | None = None,
      jit: bool = True,
      callback: _typing.CallbackFn[APDState] | None = None,
      time_limit: float | None = None,
  ) -> APDState:
    """Runs `APD` directly on any `CompositeOracleProtocol`."""
    x_proj = oracle.domain_project(tree.as_float_tree(x0))
    dtype = tree.infer_dtype(x_proj)
    L_seed = (
        jnp.asarray(self.lipschitz_init, dtype=dtype)
        if upper_curvature_init is None and self.upper_curvature_init is None
        else jnp.asarray(
            self.upper_curvature_init
            if upper_curvature_init is None
            else upper_curvature_init,
            dtype=dtype,
        )
    )

    # Step-0 refinement calibrates local smoothness M_0 and checks immediate
    # stationarity (avoiding zero-step degeneracy if x0 is already optimal).
    ref0 = certificates.refine(oracle, x_proj, L_seed, jit=jit)
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

    m_min_arr = jnp.asarray(self.lower_curvature_min, dtype=dtype)
    m_max_arr = jnp.asarray(self.lower_curvature_max, dtype=dtype)
    if lower_curvature_init is not None:
      m0 = jnp.clip(
          jnp.asarray(lower_curvature_init, dtype=dtype),
          m_min_arr,
          m_max_arr,
      )
    elif self.lower_curvature_init is not None:
      m0 = jnp.clip(
          jnp.asarray(self.lower_curvature_init, dtype=dtype),
          m_min_arr,
          m_max_arr,
      )
    elif oracle.curvature_lower is not None:
      m_known = jnp.asarray(oracle.curvature_lower, dtype=dtype)
      m0 = jnp.where(
          m_known > 0.0,
          jnp.clip(m_known, m_min_arr, m_max_arr),
          jnp.maximum(tracker0.target_opt_tol, m_min_arr),
      )
    else:
      m0 = jnp.clip(jnp.asarray(1.0, dtype=dtype), m_min_arr, m_max_arr)

    if upper_curvature_init is not None:
      M0 = jnp.maximum(
          jnp.asarray(upper_curvature_init, dtype=dtype),
          jnp.asarray(1e-8, dtype=dtype),
      )
    elif self.upper_curvature_init is not None:
      M0 = jnp.maximum(
          jnp.asarray(self.upper_curvature_init, dtype=dtype),
          jnp.asarray(1e-8, dtype=dtype),
      )
    else:
      M0 = jnp.maximum(ref0.lipschitz_used, jnp.asarray(1e-8, dtype=dtype))

    init_state = APDState(
        center=ref0.x_hat,
        center_obj=ref0.obj_val,
        lower_curvature=m0,
        upper_curvature=M0,
        never_increased_m=jnp.asarray(True, dtype=jnp.bool_),
        tracker=tracker0,
    )

    eff_rho = self._effective_rho
    pf_cfg = pf_acg.PFACGConfig(
        mu=0.5,
        sigma=eff_rho / math.sqrt(2.0),
        theta=self.theta,
        beta=self.beta,
        max_iters=self.max_inner_iters,
        lipschitz_min=0.5,
    )

    def _body(st: APDState) -> APDState:
      m_hat = st.lower_curvature
      two_m = 2.0 * m_hat
      inv_two_m = 1.0 / two_m
      sub_oracle = composite.RescaledProxOracle(
          base=oracle,
          center=st.center,
          scale=inv_two_m,
      )

      if self.decay_curvature:
        M_trial = jnp.maximum(
            jnp.asarray(1e-8, dtype=dtype),
            st.upper_curvature / (1.0 + 0.5 * self.beta),
        )
      else:
        M_trial = st.upper_curvature
      L0_sub = M_trial * inv_two_m + 1.0

      pf_st = pf_acg.run_pf_acg(
          sub_oracle,
          st.center,
          lipschitz_init=L0_sub,
          config=pf_cfg,
          two_m=two_m,
          opt_tol=st.target_opt_tol,
          norm_kind=self.norm_kind,
          jit=jit,
      )

      y_cand = pf_st.y
      phi_y = oracle.value(y_cand)
      u_cand = tree.scale(two_m, pf_st.r)
      v_cand = tree.scale(
          two_m,
          tree.add(pf_st.r, tree.sub(st.center, y_cand)),
      )
      cand_res = tree.norm(v_cand, kind=self.norm_kind)

      # Exact inclusion v_cand in grad_f(y_cand) + dh(y_cand) holds at every
      # PF.ACG iterate (Lemma 3.1(b)), so always update best certificate.
      tr_upd = st.tracker.update_incumbent(y_cand, v_cand, cand_res, phi_y)

      # Algorithm 3.3 Step 2 proximal descent checks (Eq. (3.11)).
      eps_val = tree.eps(dtype)
      eps_sq = eps_val**2
      phi_slack = 100.0 * eps_val * (1.0 + jnp.abs(st.center_obj))
      desc_ok = (
          tree.norm_sq(v_cand)
          <= 2.0 * self.theta * m_hat * (st.center_obj - phi_y + phi_slack)
          + eps_sq
      )
      dist_sq = tree.norm_sq(tree.sub(y_cand, st.center))
      prox_err_ok = (
          tree.norm_sq(u_cand)
          <= 2.0 * ((eff_rho * m_hat) ** 2) * dist_sq + eps_sq
      )
      bad_code = jnp.asarray(
          int(pf_acg.PFACGStatus.BAD_TERMINATION),
          dtype=jnp.int32,
      )
      ls_fail_code = jnp.asarray(
          int(pf_acg.PFACGStatus.LINE_SEARCH_FAILED),
          dtype=jnp.int32,
      )
      converged_now = tr_upd.best_opt_res <= st.target_opt_tol
      step_rejected = jnp.logical_and(
          jnp.logical_not(converged_now),
          jnp.logical_and(
              pf_st.status != ls_fail_code,
              jnp.logical_or(
                  pf_st.status == bad_code,
                  jnp.logical_not(jnp.logical_and(desc_ok, prox_err_ok)),
              ),
          ),
      )

      M_from_acg = jnp.maximum(
          jnp.asarray(1e-8, dtype=dtype),
          two_m * (pf_st.lipschitz - 1.0),
      )

      def _on_reject() -> APDState:
        grown_m = jnp.minimum(m_hat * self.alpha, m_max_arr)
        M_keep = jnp.maximum(st.upper_curvature, M_from_acg)
        next_tr = tr_upd.advance(
            inner_inc=pf_st.iteration,
            smooth_inc=1,
            counts_inc=pf_st.counts,
            max_iters=self.max_outer_iters,
            check_converged=False,
        )
        return APDState(
            center=st.center,
            center_obj=st.center_obj,
            lower_curvature=grown_m,
            upper_curvature=M_keep,
            never_increased_m=jnp.asarray(False, dtype=jnp.bool_),
            tracker=next_tr,
        )

      def _on_accept() -> APDState:
        # Guard float32 smoothness relaxation against tiny step norms.
        quad_curv = 0.5 * M_from_acg * dist_sq
        noise_floor = 100.0 * eps_val * (1.0 + jnp.abs(phi_y))
        M_next = jnp.where(
            quad_curv > noise_floor,
            M_from_acg,
            jnp.maximum(M_from_acg, st.upper_curvature),
        )
        if self.decay_curvature:
          m_relaxed = jnp.maximum(m_min_arr, m_hat / (1.0 + 0.5 * self.alpha))
          m_next = jnp.where(dist_sq > eps_sq, m_relaxed, m_hat)
        else:
          m_bitonic = jnp.maximum(m_min_arr, m_hat / self.alpha)
          m_next = jnp.where(st.never_increased_m, m_bitonic, m_hat)

        next_tr = tr_upd.advance(
            inner_inc=pf_st.iteration,
            smooth_inc=1,
            counts_inc=pf_st.counts,
            max_iters=self.max_outer_iters,
            line_search_failed=pf_st.status == ls_fail_code,
        )
        return APDState(
            center=y_cand,
            center_obj=phi_y,
            lower_curvature=m_next,
            upper_curvature=M_next,
            never_increased_m=st.never_increased_m,
            tracker=next_tr,
        )

      if jit:
        return typing.cast(
            APDState,
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
      callback: _typing.CallbackFn[APDState] | None = None,
      time_limit: float | None = None,
  ) -> base.Solution:
    """Solves an unconstrained `CompositeProblem` with `APD` (`PF.APD`)."""
    if problem.is_constrained:
      raise ValueError(
          "APD solves unconstrained composite problems; use QPAPD, QPAIPP, or "
          "IALM for problems with `constraints`."
      )
    return self.solve_oracle(
        problem.oracle,
        x0,
        jit=jit,
        callback=callback,
        time_limit=time_limit,
    ).to_solution()
