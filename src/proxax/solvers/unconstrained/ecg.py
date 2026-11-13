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
class ECGState(base.SolverTrackerMixin):
  """Iteration state of the `ECG` solver."""

  x: _typing.PyTree
  lipschitz: jax.Array
  lipschitz_floor: jax.Array
  tracker: base.SolverTracker


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

    init_state = ECGState(
        x=ref0.x_hat,
        lipschitz=L0,
        lipschitz_floor=L_floor,
        tracker=tracker0,
    )

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
      dist_sq = tree.norm_sq(tree.sub(x_bar, st.x))
      quad_curv = 0.5 * L_used * dist_sq
      noise_floor = 100.0 * eps_val * (1.0 + jnp.abs(f_bar))
      if is_adaptive:
        L_relaxed = jnp.maximum(st.lipschitz_floor, L_used / self.gamma_down)
        L_next = jnp.where(quad_curv > noise_floor, L_relaxed, L_used)
      else:
        L_next = L_used

      eval_inc = n_bt + 1
      tracker = st.tracker.update_incumbent(
          x_bar,
          v_bar,
          res_bar,
          obj_bar,
      ).advance(
          inner_inc=eval_inc,
          smooth_inc=eval_inc + 1,
          grad_inc=eval_inc + 1,
          prox_inc=eval_inc,
          max_iters=self.max_iters,
          line_search_failed=jnp.logical_not(ls_ok),
          finite_ok=cand_finite,
      )

      return ECGState(
          x=x_bar,
          lipschitz=L_next,
          lipschitz_floor=st.lipschitz_floor,
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
    return fin.to_solution()

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
