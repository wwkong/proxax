"""Nonconvex FISTA (`NC-FISTA`) solver (Liang, Monteiro, & Sim 2021)."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from proxax import _typing
from proxax import certificates
from proxax import problem as problem_mod
from proxax import tree
from proxax.oracle import composite
from proxax.solvers import base


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class NCFISTAState(base.SolverTrackerMixin):
  """Iteration state of the `NCFISTA` solver."""

  x: _typing.PyTree
  y: _typing.PyTree
  a_sum: jax.Array
  lam: jax.Array
  xi: jax.Array
  tracker: base.SolverTracker


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class NCFISTA:
  """Nonconvex Fast Iterative Shrinkage-Thresholding (`NC-FISTA`) solver.

  Implements `src/solvers/NC_FISTA.m` from `wwkong/nc_opt` based on J. Liang,
  R. D. C. Monteiro, and C. K. Sim, *"A FISTA-type accelerated gradient
  algorithm for solving smooth nonconvex composite optimization problems,"*
  Computational Optimization and Applications, 79(3):649--679, 2021.

  Uses fixed step size `lam` (default `0.99 / L` where `L = max(M, m)`) and
  damping parameter `xi` (`1.05 * m` when `m > 0`, reducing smoothly to standard
  convex FISTA `A_0 = 0, tau = 0` when `m = 0`). At every step, forms the exact
  composite subgradient `v = ((1 + tau) / lam) * (x_tilde - y_next) +
  grad_f(y_next) - grad_f(x_tilde)` in `grad_f(y_next) + dh(y_next)`.

  Attributes:
    opt_tol: Absolute stationarity tolerance `rho > 0`.
    rel_opt_tol: Optional relative stationarity tolerance vs step-0 norm.
    norm_kind: Residual norm (`'l2'` or `'linf'`).
    upper_curvature: Optional global upper smoothness constant `M > 0`.
    lower_curvature: Optional weak-convexity constant `m >= 0`.
    lam: Optional step size `lambda > 0` (defaults to `0.99 / max(M, m)`).
    xi: Optional nonconvex damping constant `xi > m` (defaults to `1.05 * m`
      when `m > 0` and `0.0` when `m == 0`).
    lipschitz_init: Fallback smoothness seed when `M` is unknown.
    max_iters: Maximum `NC-FISTA` iterations.
  """

  opt_tol: float = 1e-5
  rel_opt_tol: float = 0.0
  norm_kind: str = "l2"
  upper_curvature: float | None = None
  lower_curvature: float | None = None
  lam: float | None = None
  xi: float | None = None
  lipschitz_init: float = 1.0
  max_iters: int = 2000

  def solve_oracle(
      self,
      oracle: composite.CompositeOracleProtocol,
      x0: _typing.PyTree,
      *,
      opt_tol_override: _typing.Scalar | None = None,
      jit: bool = True,
      callback: _typing.CallbackFn[NCFISTAState] | None = None,
      time_limit: float | None = None,
  ) -> NCFISTAState:
    """Runs `NCFISTA` directly on any `CompositeOracleProtocol`."""
    x_proj = oracle.domain_project(tree.as_float_tree(x0))
    dtype = tree.infer_dtype(x_proj)

    if self.lower_curvature is not None:
      m_val = jnp.asarray(self.lower_curvature, dtype=dtype)
    elif oracle.curvature_lower is not None:
      m_val = jnp.asarray(oracle.curvature_lower, dtype=dtype)
    else:
      m_val = jnp.asarray(1.0, dtype=dtype)
    m_val = jnp.maximum(m_val, jnp.asarray(0.0, dtype=dtype))

    known_M = (
        self.upper_curvature is not None or oracle.curvature_upper is not None
    )
    if self.upper_curvature is not None:
      M_seed = jnp.asarray(self.upper_curvature, dtype=dtype)
    elif oracle.curvature_upper is not None:
      M_seed = jnp.asarray(oracle.curvature_upper, dtype=dtype)
    else:
      M_seed = jnp.asarray(self.lipschitz_init, dtype=dtype)

    L_seed = jnp.maximum(
        jnp.maximum(M_seed, m_val),
        jnp.asarray(1e-8, dtype=dtype),
    )
    ref0 = certificates.refine(oracle, x_proj, L_seed, jit=jit)
    L_eff = L_seed if known_M else jnp.maximum(L_seed, ref0.lipschitz_used)

    lam_arr = (
        jnp.asarray(self.lam, dtype=dtype)
        if self.lam is not None
        else 0.99 / L_eff
    )
    if self.xi is not None:
      xi_arr = jnp.asarray(self.xi, dtype=dtype)
    else:
      xi_arr = 1.05 * m_val

    has_xi = xi_arr > m_val + jnp.asarray(1e-12, dtype=dtype)
    denom = jnp.maximum((xi_arr - m_val) ** 2, jnp.asarray(1e-24, dtype=dtype))
    a_sum0 = jnp.where(
        has_xi,
        2.0 * xi_arr * (xi_arr + m_val) / denom,
        jnp.asarray(0.0, dtype=dtype),
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

    init_state = NCFISTAState(
        x=ref0.x_hat,
        y=ref0.x_hat,
        a_sum=a_sum0,
        lam=lam_arr,
        xi=xi_arr,
        tracker=tracker0,
    )

    def _body(st: NCFISTAState) -> NCFISTAState:
      a_step = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * st.a_sum))
      a_next = st.a_sum + a_step
      tau = 2.0 * st.xi * st.lam / a_step
      x_tilde = tree.convex_comb(st.a_sum, st.y, a_step, st.x)

      _, grad_tx = oracle.smooth_value_and_grad(x_tilde)
      eff_step = st.lam / (1.0 + tau)
      y_next = oracle.prox(
          tree.add_scaled(x_tilde, -eff_step, grad_tx),
          eff_step,
      )
      denom_x = a_step * (tau * a_step + 1.0)
      c_y_next = (1.0 + tau) * a_next / denom_x
      c_y_prev = st.a_sum / denom_x
      x_next = jax.tree.map(
          lambda yn, yp: c_y_next * yn - c_y_prev * yp,
          y_next,
          st.y,
      )

      f_yn, grad_yn = oracle.smooth_value_and_grad(y_next)
      v_cand = tree.add(
          tree.scale(1.0 / eff_step, tree.sub(x_tilde, y_next)),
          tree.sub(grad_yn, grad_tx),
      )
      res_cand = tree.norm(v_cand, kind=self.norm_kind)
      obj_yn = f_yn + oracle.proximal_value(y_next)

      cand_finite = jnp.logical_and(
          tree.all_finite(y_next),
          jnp.isfinite(res_cand),
      )
      tracker = st.tracker.update_incumbent(
          y_next,
          v_cand,
          res_cand,
          obj_yn,
      ).advance(
          inner_inc=1,
          smooth_inc=2,
          grad_inc=2,
          prox_inc=1,
          max_iters=self.max_iters,
          finite_ok=cand_finite,
      )

      return NCFISTAState(
          x=x_next,
          y=y_next,
          a_sum=a_next,
          lam=st.lam,
          xi=st.xi,
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
      callback: _typing.CallbackFn[NCFISTAState] | None = None,
      time_limit: float | None = None,
  ) -> base.Solution:
    """Solves an unconstrained `CompositeProblem` using `NCFISTA`."""
    if problem.is_constrained:
      raise ValueError(
          "NCFISTA solves unconstrained composite problems; pass `QPAIPP`, "
          "`QPAPD`, `IALM`, or use `proxax.solve(problem, x0)`."
      )
    fin = self.solve_oracle(
        problem.oracle,
        x0,
        jit=jit,
        callback=callback,
        time_limit=time_limit,
    )
    return fin.to_solution()
