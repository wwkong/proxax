"""Adaptive Nonconvex FISTA (`ADAP-NC-FISTA`) solver (Liang et al. 2021)."""

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
class AdapFISTAState:
  """Iteration state of the `AdapFISTA` (`ADAP-NC-FISTA`) solver."""

  z0: _typing.PyTree
  x: _typing.PyTree
  y: _typing.PyTree
  a_sum: jax.Array
  lam: jax.Array
  xi: jax.Array
  best_x: _typing.PyTree
  best_subgrad: _typing.PyTree
  best_opt_res: jax.Array
  best_obj: jax.Array
  target_opt_tol: jax.Array
  outer_iters: jax.Array
  inner_iters: jax.Array
  smooth_evals: jax.Array
  grad_evals: jax.Array
  prox_evals: jax.Array
  status: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _AdapFISTASubCarry:
  """Backtracking carry for `SUB` in `ADAP-NC-FISTA`."""

  lam: jax.Array
  xi: jax.Array
  stepsize: jax.Array
  y_next: _typing.PyTree
  f_y_next: jax.Array
  m_upper: jax.Array
  accepted: jax.Array
  count: jax.Array


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class AdapFISTA:
  """Adaptive Nonconvex FISTA (`ADAP-NC-FISTA`) solver.

  Implements `src/solvers/ADAP_FISTA.m` from `wwkong/nc_opt` based on J. Liang,
  R. D. C. Monteiro, and C. K. Sim, *"A FISTA-type accelerated gradient
  algorithm for solving smooth nonconvex composite optimization problems,"*
  Computational Optimization and Applications, 79(3):649--679, 2021.

  Dynamically adapts both the primal step size `lam` and nonconvex damping
  parameter `xi` from observed local upper/lower curvature estimates without
  requiring prior knowledge of `(m, M)`, while certifying exact composite
  stationarity `v = (x_tilde - y_next) / stepsize + grad_f(y_next) -
  grad_f(x_tilde) in grad_f(y_next) + dh(y_next)`.

  Attributes:
    opt_tol: Absolute stationarity tolerance `rho > 0`.
    rel_opt_tol: Optional relative stationarity tolerance vs step-0 norm.
    norm_kind: Residual norm (`'l2'` or `'linf'`).
    theta: Step-size contraction factor (`> 1`, default `1.25`).
    lower_curvature_init: Optional initial lower curvature `m_0 > 0`.
    upper_curvature_init: Optional initial upper curvature `M_0 > 0`.
    curv_scale: Initial optimistic curvature divisor (`M_1 = L_0 / curv_scale`).
    lipschitz_init: Seed smoothness used during step-0 calibration.
    max_iters: Maximum outer `ADAP-NC-FISTA` iterations.
    max_backtracks: Maximum `SUB` adjustments per outer iteration.
  """

  opt_tol: float = 1e-5
  rel_opt_tol: float = 0.0
  norm_kind: str = "l2"
  theta: float = 1.25
  lower_curvature_init: float | None = None
  upper_curvature_init: float | None = None
  curv_scale: float = 10.0
  lipschitz_init: float = 1.0
  max_iters: int = 2000
  max_backtracks: int = 50

  def __post_init__(self) -> None:
    """Validates `AdapFISTA` hyperparameters."""
    if self.theta <= 1.0:
      raise ValueError("`theta` must be strictly greater than 1.")
    if self.curv_scale <= 0.0:
      raise ValueError("`curv_scale` must be positive.")

  def solve_oracle(
      self,
      oracle: composite.CompositeOracleProtocol,
      x0: _typing.PyTree,
      *,
      opt_tol_override: _typing.Scalar | None = None,
      jit: bool = True,
      callback: _typing.CallbackFn[AdapFISTAState] | None = None,
      time_limit: float | None = None,
  ) -> AdapFISTAState:
    """Runs `AdapFISTA` directly on any `CompositeOracleProtocol`."""
    x_proj = oracle.domain_project(tree.as_float_tree(x0))
    dtype = tree.infer_dtype(x_proj)

    L_seed = jnp.asarray(
        self.lipschitz_init
        if self.upper_curvature_init is None
        else self.upper_curvature_init,
        dtype=dtype,
    )
    ref0 = certificates.refine(oracle, x_proj, L_seed, jit=jit)
    res0 = tree.norm(ref0.v_hat, kind=self.norm_kind)
    if opt_tol_override is not None:
      target_tol = jnp.asarray(opt_tol_override, dtype=dtype)
    else:
      target_tol = jnp.asarray(self.opt_tol, dtype=dtype) + (
          jnp.asarray(self.rel_opt_tol, dtype=dtype) * res0
      )

    if (
        self.upper_curvature_init is not None
        and self.lower_curvature_init is not None
    ):
      M0 = jnp.maximum(
          jnp.asarray(self.upper_curvature_init, dtype=dtype),
          jnp.asarray(1e-6, dtype=dtype),
      )
      m0 = jnp.maximum(
          jnp.asarray(self.lower_curvature_init, dtype=dtype),
          jnp.asarray(1e-6, dtype=dtype),
      )
    else:
      M0 = jnp.maximum(
          ref0.lipschitz_used / self.curv_scale,
          jnp.asarray(1e-4, dtype=dtype),
      )
      m0 = M0

    lam0 = 1.0 / M0
    xi0 = 2.0 * m0

    finite0 = jnp.logical_and(tree.all_finite(ref0.x_hat), jnp.isfinite(res0))
    converged0 = res0 <= target_tol
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

    init_state = AdapFISTAState(
        z0=ref0.x_hat,
        x=ref0.x_hat,
        y=ref0.x_hat,
        a_sum=jnp.asarray(0.0, dtype=dtype),
        lam=lam0,
        xi=xi0,
        best_x=ref0.x_hat,
        best_subgrad=ref0.v_hat,
        best_opt_res=res0,
        best_obj=ref0.obj_val,
        target_opt_tol=target_tol,
        outer_iters=jnp.asarray(0, dtype=jnp.int32),
        inner_iters=jnp.asarray(0, dtype=jnp.int32),
        smooth_evals=ref0.smooth_evals,
        grad_evals=ref0.grad_evals,
        prox_evals=ref0.prox_evals,
        status=status0,
    )

    def _cond(st: AdapFISTAState) -> jax.Array:
      return st.status == running_code

    def _body(st: AdapFISTAState) -> AdapFISTAState:
      eps_val = tree.eps(dtype)
      eps_sq = eps_val**2
      a_step = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * st.a_sum))
      a_next = st.a_sum + a_step
      tx = tree.convex_comb(st.a_sum, st.y, a_step, st.x)
      ty = tree.convex_comb(st.a_sum, st.y, a_step, st.z0)

      f_tx, grad_tx = oracle.smooth_value_and_grad(tx)
      diff_ty_tx = tree.sub(ty, tx)
      dist_ty_sq = tree.norm_sq(diff_ty_tx)

      def _eval_m_next() -> tuple[jax.Array, jax.Array]:
        f_ty = oracle.smooth_value(ty)
        ell_ty = f_tx + tree.vdot(grad_tx, diff_ty_tx)
        curv_low = 2.0 * (ell_ty - f_ty) / jnp.maximum(dist_ty_sq, 1e-30)
        return (
            jnp.maximum(curv_low, jnp.asarray(0.0, dtype=dtype)),
            jnp.asarray(1, dtype=jnp.int32),
        )

      m_next, extra_s = jax.lax.cond(
          jnp.logical_and(st.a_sum > 0.0, dist_ty_sq > eps_sq),
          _eval_m_next,
          lambda: (
              jnp.asarray(0.0, dtype=dtype),
              jnp.asarray(0, dtype=jnp.int32),
          ),
      )

      lam_k_init = st.lam

      def _trial_sub(
          lam_v: jax.Array,
          xi_v: jax.Array,
      ) -> tuple[jax.Array, _typing.PyTree, jax.Array, jax.Array, jax.Array]:
        stepsize = lam_v * a_step / (a_step + 2.0 * lam_v * xi_v)
        y_cand = oracle.prox(
            tree.add_scaled(tx, -stepsize, grad_tx),
            stepsize,
        )
        f_y = oracle.smooth_value(y_cand)
        diff_y = tree.sub(y_cand, tx)
        dist_y_sq = tree.norm_sq(diff_y)
        numer = (
            f_y
            - f_tx
            - tree.vdot(grad_tx, diff_y)
            - 10.0 * eps_val * (1.0 + jnp.abs(f_tx))
        )
        m_up = jnp.where(
            dist_y_sq > eps_sq,
            jnp.maximum(
                jnp.asarray(0.0, dtype=dtype),
                2.0 * numer / jnp.maximum(dist_y_sq, 1e-30),
            ),
            jnp.asarray(0.0, dtype=dtype),
        )
        cond_L = lam_v * m_up <= 0.9
        cond_xi = jnp.logical_or(
            m_next <= 0.0,
            xi_v * jnp.maximum(0.0, lam_k_init - lam_v / a_step)
            >= 0.5 * m_next * lam_v,
        )
        return stepsize, y_cand, f_y, m_up, jnp.logical_and(cond_L, cond_xi)

      ss0, y0_c, fy0, mup0, ok0 = _trial_sub(st.lam, st.xi)
      sub_init = _AdapFISTASubCarry(
          lam=st.lam,
          xi=st.xi,
          stepsize=ss0,
          y_next=y0_c,
          f_y_next=fy0,
          m_upper=mup0,
          accepted=ok0,
          count=jnp.asarray(1, dtype=jnp.int32),
      )

      def _sub_cond(c: _AdapFISTASubCarry) -> jax.Array:
        return jnp.logical_and(
            jnp.logical_not(c.accepted),
            c.count < self.max_backtracks,
        )

      def _sub_body(c: _AdapFISTASubCarry) -> _AdapFISTASubCarry:
        bad_L = c.lam * c.m_upper > 0.9
        lam_shrunk = jnp.minimum(
            c.lam / self.theta,
            0.9 / jnp.maximum(c.m_upper, 1e-30),
        )
        next_lam = jnp.where(bad_L, lam_shrunk, c.lam)
        bad_xi = jnp.logical_and(
            m_next > 0.0,
            c.xi * jnp.maximum(0.0, lam_k_init - next_lam / a_step)
            < 0.5 * m_next * next_lam,
        )
        next_xi = jnp.where(bad_xi, 2.0 * c.xi, c.xi)
        ss, yn, fy, mup, ok = _trial_sub(next_lam, next_xi)
        return _AdapFISTASubCarry(
            lam=next_lam,
            xi=next_xi,
            stepsize=ss,
            y_next=yn,
            f_y_next=fy,
            m_upper=mup,
            accepted=ok,
            count=c.count + 1,
        )

      sub_fin = _loop.while_loop(_sub_cond, _sub_body, sub_init, jit=jit)
      lam_used = sub_fin.lam
      xi_used = sub_fin.xi
      y_next = sub_fin.y_next
      stepsize = sub_fin.stepsize

      denom_x = xi_used * lam_used + 1.0
      c1 = (a_step + xi_used * lam_used) / denom_x
      c2 = (a_step - 1.0) / denom_x
      x_next = jax.tree.map(
          lambda yn, yp: c1 * yn - c2 * yp,
          y_next,
          st.y,
      )

      _, grad_yn = oracle.smooth_value_and_grad(y_next)
      v_cand = tree.add(
          tree.scale(1.0 / jnp.maximum(stepsize, 1e-30), tree.sub(tx, y_next)),
          tree.sub(grad_yn, grad_tx),
      )
      res_cand = tree.norm(v_cand, kind=self.norm_kind)
      obj_yn = sub_fin.f_y_next + oracle.proximal_value(y_next)

      cand_finite = jnp.logical_and(
          tree.all_finite(y_next),
          jnp.isfinite(res_cand),
      )
      is_better = jnp.logical_and(cand_finite, res_cand <= st.best_opt_res)
      best_x = jax.tree.map(
          lambda cy, bx: jnp.where(is_better, cy, bx),
          y_next,
          st.best_x,
      )
      best_v = jax.tree.map(
          lambda cv, bv: jnp.where(is_better, cv, bv),
          v_cand,
          st.best_subgrad,
      )
      best_res = jnp.where(is_better, res_cand, st.best_opt_res)
      best_obj = jnp.where(is_better, obj_yn, st.best_obj)

      next_outer = st.outer_iters + 1
      converged_now = best_res <= st.target_opt_tol
      next_status = jnp.where(
          jnp.logical_not(cand_finite),
          jnp.asarray(int(base.SolveStatus.NON_FINITE), dtype=jnp.int32),
          jnp.where(
              jnp.logical_not(sub_fin.accepted),
              jnp.asarray(
                  int(base.SolveStatus.LINE_SEARCH_FAILED), dtype=jnp.int32
              ),
              jnp.where(
                  converged_now,
                  jnp.asarray(int(base.SolveStatus.CONVERGED), dtype=jnp.int32),
                  jnp.where(
                      next_outer >= self.max_iters,
                      jnp.asarray(
                          int(base.SolveStatus.MAX_ITER_REACHED),
                          dtype=jnp.int32,
                      ),
                      running_code,
                  ),
              ),
          ),
      )

      return AdapFISTAState(
          z0=st.z0,
          x=x_next,
          y=y_next,
          a_sum=a_next,
          lam=lam_used,
          xi=xi_used,
          best_x=best_x,
          best_subgrad=best_v,
          best_opt_res=best_res,
          best_obj=best_obj,
          target_opt_tol=st.target_opt_tol,
          outer_iters=next_outer,
          inner_iters=st.inner_iters + sub_fin.count,
          smooth_evals=st.smooth_evals + sub_fin.count + extra_s + 2,
          grad_evals=st.grad_evals + 2,
          prox_evals=st.prox_evals + sub_fin.count,
          status=next_status,
      )

    def _on_timeout(st: AdapFISTAState) -> AdapFISTAState:
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
      callback: _typing.CallbackFn[AdapFISTAState] | None = None,
      time_limit: float | None = None,
  ) -> base.Solution:
    """Solves an unconstrained `CompositeProblem` using `AdapFISTA`."""
    if problem.is_constrained:
      raise ValueError(
          "AdapFISTA solves unconstrained composite problems; pass `QPAIPP`, "
          "`QPAPD`, `IALM`, or use `proxax.solve(problem, x0)`."
      )
    fin = self.solve_oracle(
        problem.oracle,
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
        feas_residual=jnp.asarray(0.0, dtype=fin.best_opt_res.dtype),
        obj_val=fin.best_obj,
    )
    return base.Solution(
        x=fin.best_x,
        dual=None,
        subgrad=fin.best_subgrad,
        status=fin.status,
        stats=stats,
    )
