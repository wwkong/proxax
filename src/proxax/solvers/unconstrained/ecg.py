"""Exact Composite Gradient (`ECG`) solver (Nesterov 2013)."""

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
class ECGState:
  """Iteration state of the `ECG` solver."""

  x: _typing.PyTree
  best_x: _typing.PyTree
  best_subgrad: _typing.PyTree
  best_opt_res: jax.Array
  best_obj: jax.Array
  target_opt_tol: jax.Array
  lipschitz: jax.Array
  lipschitz_floor: jax.Array
  outer_iters: jax.Array
  inner_iters: jax.Array
  smooth_evals: jax.Array
  grad_evals: jax.Array
  prox_evals: jax.Array
  status: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _ECGLineSearchCarry:
  """Internal backtracking carry for adaptive `ECG`."""

  lipschitz: jax.Array
  x_bar: _typing.PyTree
  f_bar: jax.Array
  grad_bar: _typing.PyTree
  accepted: jax.Array
  backtracks: jax.Array


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ECG:
  """Exact Composite Gradient (`ECG`) method (Nesterov 2013).

  Implements `src/solvers/ECG.m` from `wwkong/nc_opt` based on Y. Nesterov,
  *"Gradient methods for minimizing composite functions,"* Mathematical
  Programming, 140(1):125--161, 2013.

  At each step `k`, computes `x_{k+1} = prox_{h / L_k}(x_k - grad_f(x_k) / L_k)`
  (with Nesterov backtracking + relaxation when `step_type='adaptive'`) and
  forms the exact composite subgradient
  `v_{k+1} = L_k * (x_k - x_{k+1}) + grad_f(x_{k+1}) - grad_f(x_k)` in
  `grad_f(x_{k+1}) + dh(x_{k+1})`.

  Attributes:
    opt_tol: Absolute stationarity tolerance `rho > 0`.
    rel_opt_tol: Optional relative stationarity tolerance vs step-0 norm.
    norm_kind: Residual norm (`'l2'` or `'linf'`).
    step_type: Either `'adaptive'` (backtracking + relaxation) or `'constant'`.
    lipschitz_init: Initial smoothness estimate `L_0 > 0` (or fixed `L` if
      `'constant'` and oracle curvatures are unspecified).
    lipschitz_min: Minimum smoothness floor during adaptive relaxation.
    lipschitz_max: Maximum smoothness cap during backtracking.
    gamma_up: Multiplicative smoothness growth factor (`> 1`) in line search.
    gamma_down: Multiplicative smoothness decay factor (`>= 1`) after step.
    max_iters: Maximum outer composite gradient steps.
    max_backtracks: Maximum line-search doublings per outer step.
  """

  opt_tol: float = 1e-5
  rel_opt_tol: float = 0.0
  norm_kind: str = "l2"
  step_type: str = "adaptive"
  lipschitz_init: float = 1.0
  lipschitz_min: float = 1e-8
  lipschitz_max: float = 1e12
  gamma_up: float = 2.0
  gamma_down: float = 2.0
  max_iters: int = 2000
  max_backtracks: int = 40

  def __post_init__(self) -> None:
    """Validates `ECG` hyperparameters."""
    if self.step_type not in ("adaptive", "constant"):
      raise ValueError(
          f"Unknown `step_type` {self.step_type!r}; expected 'adaptive' or "
          "'constant'."
      )
    if self.gamma_up <= 1.0:
      raise ValueError("`gamma_up` must be strictly greater than 1.")
    if self.gamma_down < 1.0:
      raise ValueError("`gamma_down` must be at least 1.")

  def solve_oracle(
      self,
      oracle: composite.CompositeOracleProtocol,
      x0: _typing.PyTree,
      *,
      lipschitz_init: _typing.Scalar | None = None,
      opt_tol_override: _typing.Scalar | None = None,
      jit: bool = True,
      callback: _typing.CallbackFn[ECGState] | None = None,
      time_limit: float | None = None,
  ) -> ECGState:
    """Runs `ECG` directly on any `CompositeOracleProtocol`."""
    x_proj = oracle.domain_project(tree.as_float_tree(x0))
    dtype = tree.infer_dtype(x_proj)
    is_adaptive = self.step_type == "adaptive"

    if lipschitz_init is not None:
      L_seed = jnp.asarray(lipschitz_init, dtype=dtype)
    elif (
        not is_adaptive
        and oracle.curvature_upper is not None
        and oracle.curvature_lower is not None
    ):
      L_seed = jnp.maximum(
          jnp.asarray(oracle.curvature_upper, dtype=dtype),
          jnp.asarray(oracle.curvature_lower, dtype=dtype),
      )
    elif not is_adaptive and oracle.curvature_upper is not None:
      L_seed = jnp.asarray(oracle.curvature_upper, dtype=dtype)
    else:
      L_seed = jnp.asarray(self.lipschitz_init, dtype=dtype)

    L_seed = jnp.maximum(L_seed, jnp.asarray(self.lipschitz_min, dtype=dtype))
    ref0 = certificates.refine(oracle, x_proj, L_seed, jit=jit)
    L0 = L_seed if not is_adaptive else ref0.lipschitz_used
    L_floor = jnp.maximum(
        jnp.asarray(self.lipschitz_min, dtype=dtype),
        jnp.minimum(L_seed, L0),
    )

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

    init_state = ECGState(
        x=ref0.x_hat,
        best_x=ref0.x_hat,
        best_subgrad=ref0.v_hat,
        best_opt_res=res0,
        best_obj=ref0.obj_val,
        target_opt_tol=target_tol,
        lipschitz=L0,
        lipschitz_floor=L_floor,
        outer_iters=jnp.asarray(0, dtype=jnp.int32),
        inner_iters=jnp.asarray(0, dtype=jnp.int32),
        smooth_evals=ref0.smooth_evals,
        grad_evals=ref0.grad_evals,
        prox_evals=ref0.prox_evals,
        status=status0,
    )

    def _cond(st: ECGState) -> jax.Array:
      return st.status == running_code

    def _body(st: ECGState) -> ECGState:
      f_prev, grad_prev = oracle.smooth_value_and_grad(st.x)
      eps_val = tree.eps(dtype)

      def _trial(
          L_val: jax.Array,
      ) -> tuple[_typing.PyTree, jax.Array, _typing.PyTree, jax.Array]:
        inv_L = 1.0 / L_val
        x_bar = oracle.prox(
            tree.add_scaled(st.x, -inv_L, grad_prev),
            inv_L,
        )
        f_bar, grad_bar = oracle.smooth_value_and_grad(x_bar)
        diff = tree.sub(x_bar, st.x)
        lin_approx = f_prev + tree.vdot(grad_prev, diff)
        quad_upper = lin_approx + 0.5 * L_val * tree.norm_sq(diff)
        slack = 10.0 * eps_val * (1.0 + jnp.abs(f_prev))
        ok = (not is_adaptive) | (f_bar <= quad_upper + slack)
        return x_bar, f_bar, grad_bar, jnp.asarray(ok, dtype=jnp.bool_)

      x_b0, f_b0, g_b0, ok0 = _trial(st.lipschitz)
      if is_adaptive:
        ls_init = _ECGLineSearchCarry(
            lipschitz=st.lipschitz,
            x_bar=x_b0,
            f_bar=f_b0,
            grad_bar=g_b0,
            accepted=ok0,
            backtracks=jnp.asarray(0, dtype=jnp.int32),
        )

        def _ls_cond(c: _ECGLineSearchCarry) -> jax.Array:
          return jnp.logical_and(
              jnp.logical_not(c.accepted),
              jnp.logical_and(
                  c.backtracks < self.max_backtracks,
                  c.lipschitz < self.lipschitz_max,
              ),
          )

        def _ls_body(c: _ECGLineSearchCarry) -> _ECGLineSearchCarry:
          next_L = jnp.minimum(
              c.lipschitz * self.gamma_up,
              jnp.asarray(self.lipschitz_max, dtype=dtype),
          )
          xb, fb, gb, ok = _trial(next_L)
          return _ECGLineSearchCarry(
              lipschitz=next_L,
              x_bar=xb,
              f_bar=fb,
              grad_bar=gb,
              accepted=ok,
              backtracks=c.backtracks + 1,
          )

        ls_fin = _loop.while_loop(_ls_cond, _ls_body, ls_init, jit=jit)
        L_used = ls_fin.lipschitz
        x_bar = ls_fin.x_bar
        f_bar = ls_fin.f_bar
        grad_bar = ls_fin.grad_bar
        n_bt = ls_fin.backtracks
        ls_ok = ls_fin.accepted
      else:
        L_used = st.lipschitz
        x_bar = x_b0
        f_bar = f_b0
        grad_bar = g_b0
        n_bt = jnp.asarray(0, dtype=jnp.int32)
        ls_ok = jnp.asarray(True, dtype=jnp.bool_)

      v_bar = tree.add(
          tree.scale(L_used, tree.sub(st.x, x_bar)),
          tree.sub(grad_bar, grad_prev),
      )
      res_bar = tree.norm(v_bar, kind=self.norm_kind)
      obj_bar = f_bar + oracle.proximal_value(x_bar)

      cand_finite = jnp.logical_and(
          tree.all_finite(x_bar),
          jnp.isfinite(res_bar),
      )
      is_better = jnp.logical_and(cand_finite, res_bar <= st.best_opt_res)
      best_x = jax.tree.map(
          lambda cx, bx: jnp.where(is_better, cx, bx),
          x_bar,
          st.best_x,
      )
      best_v = jax.tree.map(
          lambda cv, bv: jnp.where(is_better, cv, bv),
          v_bar,
          st.best_subgrad,
      )
      best_res = jnp.where(is_better, res_bar, st.best_opt_res)
      best_obj = jnp.where(is_better, obj_bar, st.best_obj)

      dist_sq = tree.norm_sq(tree.sub(x_bar, st.x))
      quad_curv = 0.5 * L_used * dist_sq
      noise_floor = 100.0 * eps_val * (1.0 + jnp.abs(f_bar))
      if is_adaptive:
        L_relaxed = jnp.maximum(st.lipschitz_floor, L_used / self.gamma_down)
        L_next = jnp.where(quad_curv > noise_floor, L_relaxed, L_used)
      else:
        L_next = L_used

      next_outer = st.outer_iters + 1
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

      return ECGState(
          x=x_bar,
          best_x=best_x,
          best_subgrad=best_v,
          best_opt_res=best_res,
          best_obj=best_obj,
          target_opt_tol=st.target_opt_tol,
          lipschitz=L_next,
          lipschitz_floor=st.lipschitz_floor,
          outer_iters=next_outer,
          inner_iters=st.inner_iters + eval_inc,
          smooth_evals=st.smooth_evals + eval_inc + 1,
          grad_evals=st.grad_evals + eval_inc + 1,
          prox_evals=st.prox_evals + eval_inc,
          status=next_status,
      )

    def _on_timeout(st: ECGState) -> ECGState:
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
      callback: _typing.CallbackFn[ECGState] | None = None,
      time_limit: float | None = None,
  ) -> base.Solution:
    """Solves an unconstrained `CompositeProblem` using `ECG`."""
    if problem.is_constrained:
      raise ValueError(
          "ECG solves unconstrained composite problems; pass `QPAIPP`, "
          "`QPAPD`, `IALM`, or use `proxax.solve(problem, x0)`."
      )
    fin = self.solve_state_from_problem(
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

  def solve_state_from_problem(
      self,
      problem: problem_mod.CompositeProblem,
      x0: _typing.PyTree,
      *,
      jit: bool = True,
      callback: _typing.CallbackFn[ECGState] | None = None,
      time_limit: float | None = None,
  ) -> ECGState:
    """Runs `ECG` on an unconstrained `CompositeProblem`."""
    return self.solve_oracle(
        problem.oracle,
        x0,
        jit=jit,
        callback=callback,
        time_limit=time_limit,
    )
