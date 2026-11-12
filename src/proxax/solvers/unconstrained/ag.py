"""Nonconvex Accelerated Gradient (`AG`) solver (Ghadimi & Lan 2016)."""

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
class AGState:
  """Iteration state of the `AG` solver."""

  x: _typing.PyTree
  x_ag: _typing.PyTree
  best_x: _typing.PyTree
  best_subgrad: _typing.PyTree
  best_opt_res: jax.Array
  best_obj: jax.Array
  target_opt_tol: jax.Array
  upper_curvature: jax.Array
  outer_iters: jax.Array
  inner_iters: jax.Array
  smooth_evals: jax.Array
  grad_evals: jax.Array
  prox_evals: jax.Array
  status: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _AGLineSearchCarry:
  """Monotone upper-curvature safeguard carry when `M` is unknown."""

  upper_curvature: jax.Array
  x_ag: _typing.PyTree
  f_ag: jax.Array
  grad_ag: _typing.PyTree
  accepted: jax.Array
  backtracks: jax.Array


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class AG:
  """Nonconvex Accelerated Gradient (`AG`) method (Ghadimi & Lan 2016).

  Implements `src/solvers/AG.m` from `wwkong/nc_opt` based on Algorithm 2 of
  S. Ghadimi and G. Lan, *"Accelerated gradient methods for nonconvex nonlinear
  and stochastic programming,"* Mathematical Programming, 156(1--2):59--99,
  2016.

  With step size `beta = 1 / (2 * M)`, `alpha_k = 2 / (k + 1)`, and
  `lambda_k = k * beta / 2`, computes:

  - `x_k^{md} = (1 - alpha_k) * x_{k-1}^{ag} + alpha_k * x_{k-1}`,
  - `x_k = prox_{lambda_k * h}(x_{k-1} - lambda_k * grad_f(x_k^{md}))`,
  - `x_k^{ag} = prox_{beta * h}(x_k^{md} - beta * grad_f(x_k^{md}))`,

  and certifies exact composite stationarity at `x_k^{ag}` via
  `v_k = grad_f(x_k^{ag}) - grad_f(x_k^{md}) + (x_k^{md} - x_k^{ag}) / beta`
  in `grad_f(x_k^{ag}) + dh(x_k^{ag})`. When `M` is not provided on `AG` or the
  oracle, `AG` automatically safeguards `M` monotonically via backtracking.

  Attributes:
    opt_tol: Absolute stationarity tolerance `rho > 0`.
    rel_opt_tol: Optional relative stationarity tolerance vs step-0 norm.
    norm_kind: Residual norm (`'l2'` or `'linf'`).
    upper_curvature: Optional global upper smoothness constant `M > 0`.
    lipschitz_init: Initial smoothness seed when `M` is unknown.
    lipschitz_max: Maximum allowed smoothness estimate when adapting `M`.
    max_iters: Maximum `AG` iterations.
    max_backtracks: Maximum backtracking doublings per step when `M` is unknown.
  """

  opt_tol: float = 1e-5
  rel_opt_tol: float = 0.0
  norm_kind: str = "l2"
  upper_curvature: float | None = None
  lipschitz_init: float = 1.0
  lipschitz_max: float = 1e12
  max_iters: int = 2000
  max_backtracks: int = 40

  def solve_oracle(
      self,
      oracle: composite.CompositeOracleProtocol,
      x0: _typing.PyTree,
      *,
      upper_curvature_init: _typing.Scalar | None = None,
      opt_tol_override: _typing.Scalar | None = None,
      jit: bool = True,
      callback: _typing.CallbackFn[AGState] | None = None,
      time_limit: float | None = None,
  ) -> AGState:
    """Runs `AG` directly on any `CompositeOracleProtocol`."""
    x_proj = oracle.domain_project(tree.as_float_tree(x0))
    dtype = tree.infer_dtype(x_proj)

    known_M = (
        upper_curvature_init is not None
        or self.upper_curvature is not None
        or oracle.curvature_upper is not None
    )
    if upper_curvature_init is not None:
      M_seed = jnp.asarray(upper_curvature_init, dtype=dtype)
    elif self.upper_curvature is not None:
      M_seed = jnp.asarray(self.upper_curvature, dtype=dtype)
    elif oracle.curvature_upper is not None:
      M_seed = jnp.asarray(oracle.curvature_upper, dtype=dtype)
    else:
      M_seed = jnp.asarray(self.lipschitz_init, dtype=dtype)
    M_seed = jnp.maximum(M_seed, jnp.asarray(1e-8, dtype=dtype))

    ref0 = certificates.refine(oracle, x_proj, M_seed, jit=jit)
    M0 = M_seed if known_M else jnp.maximum(M_seed, ref0.lipschitz_used)
    res0 = tree.norm(ref0.v_hat, kind=self.norm_kind)
    if opt_tol_override is not None:
      target_tol = jnp.asarray(opt_tol_override, dtype=dtype)
    else:
      target_tol = jnp.asarray(self.opt_tol, dtype=dtype) + (
          jnp.asarray(self.rel_opt_tol, dtype=dtype) * res0
      )

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

    init_state = AGState(
        x=ref0.x_hat,
        x_ag=ref0.x_hat,
        best_x=ref0.x_hat,
        best_subgrad=ref0.v_hat,
        best_opt_res=res0,
        best_obj=ref0.obj_val,
        target_opt_tol=target_tol,
        upper_curvature=M0,
        outer_iters=jnp.asarray(0, dtype=jnp.int32),
        inner_iters=jnp.asarray(0, dtype=jnp.int32),
        smooth_evals=ref0.smooth_evals,
        grad_evals=ref0.grad_evals,
        prox_evals=ref0.prox_evals,
        status=status0,
    )

    def _cond(st: AGState) -> jax.Array:
      return st.status == running_code

    def _body(st: AGState) -> AGState:
      k = st.outer_iters + 1
      k_f = k.astype(dtype)
      alpha_k = 2.0 / (k_f + 1.0)
      x_md = jax.tree.map(
          lambda xag, xp: (1.0 - alpha_k) * xag + alpha_k * xp,
          st.x_ag,
          st.x,
      )
      f_md, grad_md = oracle.smooth_value_and_grad(x_md)
      eps_val = tree.eps(dtype)

      def _trial_ag(
          M_val: jax.Array,
      ) -> tuple[_typing.PyTree, jax.Array, _typing.PyTree, jax.Array]:
        beta = 0.5 / M_val
        x_ag_cand = oracle.prox(
            tree.add_scaled(x_md, -beta, grad_md),
            beta,
        )
        f_ag, grad_ag = oracle.smooth_value_and_grad(x_ag_cand)
        diff = tree.sub(x_ag_cand, x_md)
        quad_upper = (
            f_md + tree.vdot(grad_md, diff) + 0.5 * M_val * tree.norm_sq(diff)
        )
        slack = 10.0 * eps_val * (1.0 + jnp.abs(f_md))
        ok = known_M | (f_ag <= quad_upper + slack)
        return x_ag_cand, f_ag, grad_ag, jnp.asarray(ok, dtype=jnp.bool_)

      x_ag0, f_ag0, g_ag0, ok0 = _trial_ag(st.upper_curvature)
      if not known_M:
        ls_init = _AGLineSearchCarry(
            upper_curvature=st.upper_curvature,
            x_ag=x_ag0,
            f_ag=f_ag0,
            grad_ag=g_ag0,
            accepted=ok0,
            backtracks=jnp.asarray(0, dtype=jnp.int32),
        )

        def _ls_cond(c: _AGLineSearchCarry) -> jax.Array:
          return jnp.logical_and(
              jnp.logical_not(c.accepted),
              jnp.logical_and(
                  c.backtracks < self.max_backtracks,
                  c.upper_curvature < self.lipschitz_max,
              ),
          )

        def _ls_body(c: _AGLineSearchCarry) -> _AGLineSearchCarry:
          next_M = jnp.minimum(
              2.0 * c.upper_curvature,
              jnp.asarray(self.lipschitz_max, dtype=dtype),
          )
          xag, fag, gag, ok = _trial_ag(next_M)
          return _AGLineSearchCarry(
              upper_curvature=next_M,
              x_ag=xag,
              f_ag=fag,
              grad_ag=gag,
              accepted=ok,
              backtracks=c.backtracks + 1,
          )

        ls_fin = _loop.while_loop(_ls_cond, _ls_body, ls_init, jit=jit)
        M_used = ls_fin.upper_curvature
        x_ag_next = ls_fin.x_ag
        f_ag_next = ls_fin.f_ag
        grad_ag_next = ls_fin.grad_ag
        n_bt = ls_fin.backtracks
        ls_ok = ls_fin.accepted
      else:
        M_used = st.upper_curvature
        x_ag_next = x_ag0
        f_ag_next = f_ag0
        grad_ag_next = g_ag0
        n_bt = jnp.asarray(0, dtype=jnp.int32)
        ls_ok = jnp.asarray(True, dtype=jnp.bool_)

      beta_used = 0.5 / M_used
      lam_k = 0.5 * k_f * beta_used
      x_next = oracle.prox(
          tree.add_scaled(st.x, -lam_k, grad_md),
          lam_k,
      )

      v_bar = tree.add(
          tree.sub(grad_ag_next, grad_md),
          tree.scale(1.0 / beta_used, tree.sub(x_md, x_ag_next)),
      )
      res_bar = tree.norm(v_bar, kind=self.norm_kind)
      obj_ag = f_ag_next + oracle.proximal_value(x_ag_next)

      cand_finite = jnp.logical_and(
          tree.all_finite(x_ag_next),
          jnp.isfinite(res_bar),
      )
      is_better = jnp.logical_and(cand_finite, res_bar <= st.best_opt_res)
      best_x = jax.tree.map(
          lambda cx, bx: jnp.where(is_better, cx, bx),
          x_ag_next,
          st.best_x,
      )
      best_v = jax.tree.map(
          lambda cv, bv: jnp.where(is_better, cv, bv),
          v_bar,
          st.best_subgrad,
      )
      best_res = jnp.where(is_better, res_bar, st.best_opt_res)
      best_obj = jnp.where(is_better, obj_ag, st.best_obj)

      eval_inc = n_bt + 1
      converged_now = best_res <= st.target_opt_tol
      next_status = jnp.where(
          jnp.logical_not(cand_finite),
          jnp.asarray(int(base.SolveStatus.NON_FINITE), dtype=jnp.int32),
          jnp.where(
              jnp.logical_not(ls_ok),
              jnp.asarray(
                  int(base.SolveStatus.LINE_SEARCH_FAILED), dtype=jnp.int32
              ),
              jnp.where(
                  converged_now,
                  jnp.asarray(int(base.SolveStatus.CONVERGED), dtype=jnp.int32),
                  jnp.where(
                      k >= self.max_iters,
                      jnp.asarray(
                          int(base.SolveStatus.MAX_ITER_REACHED),
                          dtype=jnp.int32,
                      ),
                      running_code,
                  ),
              ),
          ),
      )

      return AGState(
          x=x_next,
          x_ag=x_ag_next,
          best_x=best_x,
          best_subgrad=best_v,
          best_opt_res=best_res,
          best_obj=best_obj,
          target_opt_tol=st.target_opt_tol,
          upper_curvature=M_used,
          outer_iters=k,
          inner_iters=st.inner_iters + eval_inc,
          smooth_evals=st.smooth_evals + eval_inc + 1,
          grad_evals=st.grad_evals + eval_inc + 1,
          prox_evals=st.prox_evals + eval_inc + 1,
          status=next_status,
      )

    def _on_timeout(st: AGState) -> AGState:
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
      callback: _typing.CallbackFn[AGState] | None = None,
      time_limit: float | None = None,
  ) -> base.Solution:
    """Solves an unconstrained `CompositeProblem` using `AG`."""
    if problem.is_constrained:
      raise ValueError(
          "AG solves unconstrained composite problems; pass `QPAIPP`, "
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
