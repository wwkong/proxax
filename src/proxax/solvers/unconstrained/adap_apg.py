"""Adaptive Accelerated Proximal Gradient (`AdapAPG`, Lin & Xiao 2014)."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from proxax import _loop
from proxax import _typing
from proxax import certificates
from proxax import problem as problem_mod
from proxax import tree
from proxax.oracle import composite
from proxax.solvers import base


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class AdapAPGState(base.SolverTrackerMixin):
  """Iteration state of the `AdapAPG` solver."""

  x: _typing.PyTree
  x_prev: _typing.PyTree
  alpha: jax.Array
  theta: jax.Array
  lipschitz_est: jax.Array
  cur_obj: jax.Array
  tracker: base.SolverTracker


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _ALSCarry:
  """Line-search carry for `AccelLineSearch` inside `AdapAPG`."""

  m_cur: jax.Array
  alpha: jax.Array
  theta: jax.Array
  tx: _typing.PyTree
  grad_tx: _typing.PyTree
  x_p1: _typing.PyTree
  f_xp1: jax.Array
  accepted: jax.Array
  count: jax.Array


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class AdapAPG:
  """Adaptive Accelerated Proximal Gradient (`AdapAPG`) solver.

  Implements `src/solvers/AdapAPG.m` from `wwkong/nc_opt` based on Q. Lin and
  L. Xiao, *"An adaptive accelerated proximal gradient method and its homotopy
  continuation for sparse optimization,"* ICML, PMLR 32(1):73--81, 2014.

  Supports strongly convex (`mu > 0`, using `alpha_k = sqrt(mu / M_k)`),
  general convex (`mu = 0`, solving `M_k alpha_k^2 = (1 - alpha_k) theta`),
  and nonconvex composite problems (via optional monotone safeguard restarts),
  while certifying exact composite stationarity `v_k = M_k (tx_k - x_{k+1}) +
  grad_f(x_{k+1}) - grad_f(tx_k) in grad_f(x_{k+1}) + dh(x_{k+1})`.

  Attributes:
    opt_tol: Absolute stationarity tolerance `rho > 0`.
    rel_opt_tol: Optional relative stationarity tolerance vs step-0 norm.
    norm_kind: Residual norm (`'l2'` or `'linf'`).
    strong_convexity: Optional strong convexity `mu >= 0`. Defaults to
      `oracle.strong_convexity`.
    step_type: `'variable'` (adaptive increase/decrease) or `'constant'`.
    gamma_up: Line-search growth factor `gamma_1 > 1` (default `2.0`).
    gamma_down: Smoothness relaxation divisor `gamma_2 >= 1` (default `2.0`).
    monotone: If True, restarts momentum from a descent proximal-gradient step
      whenever an extrapolated step increases the composite objective.
    lipschitz_init: Initial smoothness estimate `L_0 > 0`.
    lipschitz_min: Minimum allowed smoothness estimate `L_min > 0`.
    lipschitz_max: Maximum allowed smoothness estimate `L_max`.
    max_iters: Maximum outer `AdapAPG` iterations.
    max_backtracks: Maximum backtracking trials per iteration.
  """

  opt_tol: float = 1e-5
  rel_opt_tol: float = 0.0
  norm_kind: str = "l2"
  strong_convexity: float | None = None
  step_type: str = "variable"
  gamma_up: float = 2.0
  gamma_down: float = 2.0
  monotone: bool = True
  lipschitz_init: float = 1.0
  lipschitz_min: float = 1e-6
  lipschitz_max: float = 1e12
  max_iters: int = 2000
  max_backtracks: int = 40

  def __post_init__(self) -> None:
    """Validates `AdapAPG` hyperparameters."""
    if self.step_type not in ("variable", "constant"):
      raise ValueError(
          "`step_type` must be 'variable' or 'constant'; "
          f"got {self.step_type!r}."
      )
    if self.step_type == "variable" and self.gamma_up <= 1.0:
      raise ValueError("`gamma_up` must be > 1 when `step_type='variable'`.")
    if self.gamma_down < 1.0:
      raise ValueError("`gamma_down` must be >= 1.")
    if self.lipschitz_min <= 0.0:
      raise ValueError("`lipschitz_min` must be positive.")

  def solve_oracle(
      self,
      oracle: composite.CompositeOracleProtocol,
      x0: _typing.PyTree,
      *,
      opt_tol_override: _typing.Scalar | None = None,
      jit: bool = True,
      callback: _typing.CallbackFn[AdapAPGState] | None = None,
      time_limit: float | None = None,
  ) -> AdapAPGState:
    """Runs `AdapAPG` directly on any `CompositeOracleProtocol`."""
    x_proj = oracle.domain_project(tree.as_float_tree(x0))
    dtype = tree.infer_dtype(x_proj)

    mu_smooth = (
        jnp.asarray(self.strong_convexity, dtype=dtype)
        if self.strong_convexity is not None
        else jnp.asarray(0.0, dtype=dtype)
    )
    mu_smooth = jnp.maximum(mu_smooth, jnp.asarray(0.0, dtype=dtype))
    mu_prox = jnp.maximum(
        jnp.asarray(oracle.strong_convexity, dtype=dtype),
        jnp.asarray(0.0, dtype=dtype),
    )
    mu_val = mu_smooth + mu_prox

    known_L = oracle.curvature_upper
    if self.step_type == "constant" and known_L is not None:
      L_seed = jnp.asarray(known_L, dtype=dtype)
    else:
      L_seed = jnp.asarray(self.lipschitz_init, dtype=dtype)
    L_seed = jnp.maximum(
        L_seed,
        jnp.maximum(jnp.asarray(self.lipschitz_min, dtype=dtype), mu_smooth),
    )

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

    L0 = jnp.maximum(
        ref0.lipschitz_used,
        jnp.maximum(jnp.asarray(self.lipschitz_min, dtype=dtype), mu_smooth),
    )
    theta0 = jnp.where(mu_val > 0.0, mu_val, L0)

    init_state = AdapAPGState(
        x=ref0.x_hat,
        x_prev=ref0.x_hat,
        alpha=jnp.asarray(1.0, dtype=dtype),
        theta=theta0,
        lipschitz_est=L0,
        cur_obj=ref0.obj_val,
        tracker=tracker0,
    )

    def _body(st: AdapAPGState) -> AdapAPGState:
      eps_val = tree.eps(dtype)
      l_min = jnp.maximum(
          jnp.asarray(self.lipschitz_min, dtype=dtype),
          mu_smooth,
      )
      l_max = jnp.asarray(self.lipschitz_max, dtype=dtype)

      def _als_trial(
          m_cand: jax.Array,
      ) -> tuple[
          jax.Array,
          jax.Array,
          jax.Array,
          _typing.PyTree,
          _typing.PyTree,
          _typing.PyTree,
          jax.Array,
          jax.Array,
      ]:
        m_eff = jnp.maximum(m_cand, l_min)
        alpha_sc = jnp.sqrt(mu_val / jnp.maximum(m_eff + mu_prox, 1e-30))
        beta_sc = (
            alpha_sc
            * (1.0 - st.alpha)
            / jnp.maximum(st.alpha * (1.0 + alpha_sc), 1e-30)
        )
        alpha_cvx = (
            -st.theta + jnp.sqrt(st.theta**2 + 4.0 * m_eff * st.theta)
        ) / (2.0 * m_eff)
        beta_cvx = (
            alpha_cvx
            * (1.0 - st.alpha)
            / jnp.maximum(st.alpha**2 + alpha_cvx, 1e-30)
        )

        use_sc = mu_val > 0.0
        alpha_c = jnp.where(use_sc, alpha_sc, alpha_cvx)
        beta_c = jnp.where(use_sc, beta_sc, beta_cvx)
        theta_c = jnp.where(use_sc, mu_val, m_eff * alpha_c**2)

        tx_c = jax.tree.map(
            lambda xc, xp: xc + beta_c * (xc - xp),
            st.x,
            st.x_prev,
        )
        f_tx, g_tx = oracle.smooth_value_and_grad(tx_c)
        inv_m = 1.0 / m_eff
        x_p1 = oracle.prox(tree.add_scaled(tx_c, -inv_m, g_tx), inv_m)
        f_xp1 = oracle.smooth_value(x_p1)

        if self.step_type == "constant":
          ok = jnp.asarray(True)
        else:
          diff = tree.sub(x_p1, tx_c)
          quad_upper = (
              f_tx + tree.vdot(g_tx, diff) + 0.5 * m_eff * tree.norm_sq(diff)
          )
          slack = 10.0 * eps_val * (1.0 + jnp.abs(f_tx))
          ok = f_xp1 <= quad_upper + slack
        return m_eff, alpha_c, theta_c, tx_c, g_tx, x_p1, f_xp1, ok

      m0, al0, th0, tx0, gtx0, xp0, fxp0, ok0 = _als_trial(st.lipschitz_est)
      als_init = _ALSCarry(
          m_cur=m0,
          alpha=al0,
          theta=th0,
          tx=tx0,
          grad_tx=gtx0,
          x_p1=xp0,
          f_xp1=fxp0,
          accepted=ok0,
          count=jnp.asarray(1, dtype=jnp.int32),
      )

      def _als_cond(c: _ALSCarry) -> jax.Array:
        return jnp.logical_and(
            jnp.logical_not(c.accepted),
            jnp.logical_and(c.count < self.max_backtracks, c.m_cur < l_max),
        )

      def _als_body(c: _ALSCarry) -> _ALSCarry:
        next_m_in = jnp.minimum(c.m_cur * self.gamma_up, l_max)
        mc, alc, thc, txc, gtxc, xpc, fxpc, okc = _als_trial(next_m_in)
        return _ALSCarry(
            m_cur=mc,
            alpha=alc,
            theta=thc,
            tx=txc,
            grad_tx=gtxc,
            x_p1=xpc,
            f_xp1=fxpc,
            accepted=okc,
            count=c.count + 1,
        )

      als_fin = _loop.while_loop(_als_cond, _als_body, als_init, jit=jit)
      obj_als = als_fin.f_xp1 + oracle.proximal_value(als_fin.x_p1)

      def _accept_als() -> tuple[
          _typing.PyTree,
          _typing.PyTree,
          jax.Array,
          jax.Array,
          jax.Array,
          jax.Array,
          _typing.PyTree,
          jax.Array,
          jax.Array,
          jax.Array,
      ]:
        _, g_xp1 = oracle.smooth_value_and_grad(als_fin.x_p1)
        v_als = tree.add(
            tree.scale(als_fin.m_cur, tree.sub(als_fin.tx, als_fin.x_p1)),
            tree.sub(g_xp1, als_fin.grad_tx),
        )
        return (
            als_fin.x_p1,
            st.x,
            als_fin.alpha,
            als_fin.theta,
            als_fin.m_cur,
            obj_als,
            v_als,
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )

      def _restart_pg() -> tuple[
          _typing.PyTree,
          _typing.PyTree,
          jax.Array,
          jax.Array,
          jax.Array,
          jax.Array,
          _typing.PyTree,
          jax.Array,
          jax.Array,
          jax.Array,
      ]:
        ref_r = certificates.refine(oracle, st.x, als_fin.m_cur, jit=jit)
        m_r = jnp.maximum(ref_r.lipschitz_used, l_min)
        th_r = jnp.where(mu_val > 0.0, mu_val, m_r)
        return (
            ref_r.x_hat,
            ref_r.x_hat,
            jnp.asarray(1.0, dtype=dtype),
            th_r,
            m_r,
            ref_r.obj_val,
            ref_r.v_hat,
            ref_r.smooth_evals,
            ref_r.grad_evals,
            ref_r.prox_evals,
        )

      if self.monotone:
        obj_slack = 10.0 * eps_val * (1.0 + jnp.abs(st.cur_obj))
        do_restart = jnp.logical_and(
            st.alpha < 1.0,
            obj_als > st.cur_obj + obj_slack,
        )
        (
            x_next,
            x_prev_next,
            alpha_next,
            theta_next,
            m_used,
            obj_next,
            v_cand,
            extra_s,
            extra_g,
            extra_p,
        ) = jax.lax.cond(do_restart, _restart_pg, _accept_als)
      else:
        (
            x_next,
            x_prev_next,
            alpha_next,
            theta_next,
            m_used,
            obj_next,
            v_cand,
            extra_s,
            extra_g,
            extra_p,
        ) = _accept_als()

      if self.step_type == "constant":
        l_next = m_used
      else:
        quad_curv = 0.5 * als_fin.m_cur * tree.norm_sq(
            tree.sub(als_fin.x_p1, als_fin.tx)
        )
        noise_floor = 100.0 * eps_val * (1.0 + jnp.abs(als_fin.f_xp1))
        relaxed_l = jnp.maximum(l_min, m_used / self.gamma_down)
        l_next = jnp.where(quad_curv > noise_floor, relaxed_l, m_used)

      res_cand = tree.norm(v_cand, kind=self.norm_kind)
      cand_finite = jnp.logical_and(
          tree.all_finite(x_next),
          jnp.isfinite(res_cand),
      )
      next_tr = st.tracker.update_incumbent(
          x_next, v_cand, res_cand, obj_next
      ).advance(
          inner_inc=als_fin.count,
          smooth_inc=2 * als_fin.count + extra_s,
          grad_inc=als_fin.count + extra_g,
          prox_inc=als_fin.count + extra_p,
          max_iters=self.max_iters,
          finite_ok=cand_finite,
      )

      return AdapAPGState(
          x=x_next,
          x_prev=x_prev_next,
          alpha=alpha_next,
          theta=theta_next,
          lipschitz_est=l_next,
          cur_obj=obj_next,
          tracker=next_tr,
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
      callback: _typing.CallbackFn[AdapAPGState] | None = None,
      time_limit: float | None = None,
  ) -> base.Solution:
    """Solves an unconstrained `CompositeProblem` using `AdapAPG`."""
    if problem.is_constrained:
      raise ValueError(
          "AdapAPG solves unconstrained composite problems; pass `QPAIPP`, "
          "`QPAPD`, `IALM`, or use `proxax.solve(problem, x0)`."
      )
    return self.solve_oracle(
        problem.oracle,
        x0,
        jit=jit,
        callback=callback,
        time_limit=time_limit,
    ).to_solution()
