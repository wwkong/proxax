"""Unified Problem-Parameter-Free Accelerated Gradient (`UPFAG`) solver."""

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
class UPFAGState(base.SolverTrackerMixin):
  """Iteration state of the `UPFAG` solver."""

  x_k: _typing.PyTree
  x_ag: _typing.PyTree
  x_ag_old: _typing.PyTree
  x_md_prev: _typing.PyTree
  x_ag_tilde_prev: _typing.PyTree
  sum_lambda: jax.Array
  lambda0: jax.Array
  beta0: jax.Array
  tracker: base.SolverTracker


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _UPFAGCvxCarry:
  """Convex accelerated line-search carry (Eqs. (2.2)--(2.6))."""

  itr: jax.Array
  lambda_k: jax.Array
  alpha_k: jax.Array
  x_md: _typing.PyTree
  grad_md: _typing.PyTree
  x_k_next: _typing.PyTree
  x_ag_tilde: _typing.PyTree
  phi_ag_tilde: jax.Array
  accepted: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _UPFAGNocvxCarry:
  """Nonconvex proximal gradient line-search carry (Eqs. (2.8)--(2.10))."""

  itr: jax.Array
  beta_k: jax.Array
  x_ag_bar: _typing.PyTree
  f_ag_bar: jax.Array
  grad_ag_bar: _typing.PyTree
  phi_ag_bar: jax.Array
  accepted: jax.Array


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class UPFAG:
  """Unified Problem-Parameter-Free Accelerated Gradient (`UPFAG`) method.

  Implements `src/solvers/UPFAG.m` from `wwkong/nc_opt` based on S. Ghadimi,
  G. Lan, and H. Zhang, *"Generalized uniformly optimal methods for nonlinear
  programming,"* Journal of Scientific Computing, 79(3):1854--1881, 2019.

  Combines an accelerated convex line search (producing `x_ag_tilde`) with a
  nonconvex composite gradient line search (producing `x_ag_bar` and exact
  certificate `v_k in grad_f(x_ag_bar) + dh(x_ag_bar)`), then sets the next
  aggregate point `x_ag` to the best objective value among
  `{x_ag, x_ag_bar, x_ag_tilde}`. Supports both `'monotonic'` and Barzilai-
  Borwein (`'bb'`) initial step sizes.

  Attributes:
    opt_tol: Absolute stationarity tolerance `rho > 0`.
    rel_opt_tol: Optional relative stationarity tolerance vs step-0 norm.
    norm_kind: Residual norm (`'l2'` or `'linf'`).
    line_search_type: Either `'monotonic'` or `'bb'` (Barzilai-Borwein).
    gamma_1: Contraction factor in `(0, 1)` for the convex line search.
    gamma_2: Contraction factor in `(0, 1)` for the nonconvex line search.
    gamma_3: Nonnegative Hölder slack coefficient `gamma_3 / (k + 1)` in Eq.
      (2.9) (`0.0` for standard Lipschitz-gradient problems).
    delta: Convex line-search slack `delta * alpha_k` in Eq. (2.6).
    sigma_min: Positive lower bound on Barzilai-Borwein trial step sizes.
    lipschitz_init: Initial smoothness seed `M_0 > 0` (`lambda_0 = 1 / M_0`).
    max_iters: Maximum outer `UPFAG` iterations.
    max_backtracks: Maximum backtracking trials per line search.
  """

  opt_tol: float = 1e-5
  rel_opt_tol: float = 0.0
  norm_kind: str = "l2"
  line_search_type: str = "monotonic"
  gamma_1: float = 0.4
  gamma_2: float = 0.4
  gamma_3: float = 0.0
  delta: float = 1e-2
  sigma_min: float = 1e-10
  lipschitz_init: float = 1.0
  max_iters: int = 1000
  max_backtracks: int = 40

  def __post_init__(self) -> None:
    """Validates `UPFAG` hyperparameters."""
    if self.line_search_type not in ("monotonic", "bb"):
      raise ValueError(
          f"Unknown `line_search_type` {self.line_search_type!r}; expected "
          "'monotonic' or 'bb'."
      )
    if not (0.0 < self.gamma_1 < 1.0):
      raise ValueError("`gamma_1` must lie in `(0, 1)`.")
    if not (0.0 < self.gamma_2 < 1.0):
      raise ValueError("`gamma_2` must lie in `(0, 1)`.")
    if self.gamma_3 < 0.0:
      raise ValueError("`gamma_3` must be non-negative.")

  def solve_oracle(
      self,
      oracle: composite.CompositeOracleProtocol,
      x0: _typing.PyTree,
      *,
      lipschitz_init: _typing.Scalar | None = None,
      opt_tol_override: _typing.Scalar | None = None,
      jit: bool = True,
      callback: _typing.CallbackFn[UPFAGState] | None = None,
      time_limit: float | None = None,
  ) -> UPFAGState:
    """Runs `UPFAG` directly on any `CompositeOracleProtocol`."""
    x_proj = oracle.domain_project(tree.as_float_tree(x0))
    dtype = tree.infer_dtype(x_proj)
    use_bb = self.line_search_type == "bb"

    if lipschitz_init is not None:
      M_seed = jnp.asarray(lipschitz_init, dtype=dtype)
    elif oracle.curvature_upper is not None:
      M_seed = jnp.asarray(oracle.curvature_upper, dtype=dtype)
      if oracle.curvature_lower is not None:
        M_seed = jnp.maximum(
            M_seed,
            jnp.asarray(oracle.curvature_lower, dtype=dtype),
        )
    else:
      M_seed = jnp.asarray(self.lipschitz_init, dtype=dtype)
    M_seed = jnp.maximum(M_seed, jnp.asarray(1e-8, dtype=dtype))

    ref0 = certificates.refine(oracle, x_proj, M_seed, jit=jit)
    M0 = jnp.maximum(M_seed, ref0.lipschitz_used)
    step0 = jnp.maximum(1.0 / M0, jnp.asarray(self.sigma_min, dtype=dtype))

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

    init_state = UPFAGState(
        x_k=ref0.x_hat,
        x_ag=ref0.x_hat,
        x_ag_old=ref0.x_hat,
        x_md_prev=ref0.x_hat,
        x_ag_tilde_prev=ref0.x_hat,
        sum_lambda=jnp.asarray(0.0, dtype=dtype),
        lambda0=step0,
        beta0=step0,
        tracker=tracker0,
    )

    def _body(st: UPFAGState) -> UPFAGState:
      eps_val = tree.eps(dtype)
      sigma_arr = jnp.asarray(self.sigma_min, dtype=dtype)
      extra_grads = jnp.asarray(0, dtype=jnp.int32)

      if use_bb:

        def _bb_cvx() -> tuple[jax.Array, jax.Array]:
          sk = tree.sub(st.x_ag_tilde_prev, st.x_md_prev)
          _, g_tilde = oracle.smooth_value_and_grad(st.x_ag_tilde_prev)
          _, g_md_p = oracle.smooth_value_and_grad(st.x_md_prev)
          yk = tree.sub(g_tilde, g_md_p)
          sty = tree.vdot(sk, yk)
          lam_bb = jnp.where(
              sty > 0.0,
              jnp.maximum(
                  tree.norm_sq(sk) / jnp.maximum(sty, 1e-30), sigma_arr
              ),
              jnp.maximum(st.lambda0, sigma_arr),
          )
          return lam_bb, jnp.asarray(2, dtype=jnp.int32)

        lambda_bb, bb_g1 = jax.lax.cond(
            st.outer_iters > 0,
            _bb_cvx,
            lambda: (
                jnp.maximum(st.lambda0, sigma_arr),
                jnp.asarray(0, dtype=jnp.int32),
            ),
        )
        extra_grads = extra_grads + bb_g1
      else:
        lambda_bb = jnp.maximum(st.lambda0, sigma_arr)

      def _trial_cvx(
          itr_i: jax.Array,
      ) -> tuple[
          jax.Array,
          jax.Array,
          _typing.PyTree,
          _typing.PyTree,
          _typing.PyTree,
          _typing.PyTree,
          jax.Array,
          jax.Array,
      ]:
        eta_k = lambda_bb * (self.gamma_1 ** itr_i.astype(dtype))
        lam_k = (
            0.5
            * eta_k
            * (
                1.0
                + jnp.sqrt(
                    1.0 + 4.0 * st.sum_lambda / jnp.maximum(eta_k, 1e-30)
                )
            )
        )
        alpha_k = lam_k / (lam_k + st.sum_lambda)
        x_md = jax.tree.map(
            lambda xag, xk: (1.0 - alpha_k) * xag + alpha_k * xk,
            st.x_ag,
            st.x_k,
        )
        f_md, grad_md = oracle.smooth_value_and_grad(x_md)
        phi_md = f_md + oracle.proximal_value(x_md)
        x_k_cand = oracle.prox(
            tree.add_scaled(st.x_k, -lam_k, grad_md),
            lam_k,
        )
        x_ag_tilde = jax.tree.map(
            lambda xag, xkc: (1.0 - alpha_k) * xag + alpha_k * xkc,
            st.x_ag,
            x_k_cand,
        )
        phi_tilde = oracle.value(x_ag_tilde)
        diff_md = tree.sub(x_ag_tilde, x_md)
        rhs = (
            phi_md
            + tree.vdot(grad_md, diff_md)
            + tree.norm_sq(diff_md) / jnp.maximum(2.0 * alpha_k * lam_k, 1e-30)
            + self.delta * alpha_k
            + 10.0 * eps_val * (1.0 + jnp.abs(phi_md))
        )
        return (
            lam_k,
            alpha_k,
            x_md,
            grad_md,
            x_k_cand,
            x_ag_tilde,
            phi_tilde,
            phi_tilde <= rhs,
        )

      it0 = jnp.asarray(0, dtype=jnp.int32)
      lam0_c, alp0_c, xmd0, gmd0, xk0, xagt0, phit0, ok_c0 = _trial_cvx(it0)
      cvx_init = _UPFAGCvxCarry(
          itr=it0,
          lambda_k=lam0_c,
          alpha_k=alp0_c,
          x_md=xmd0,
          grad_md=gmd0,
          x_k_next=xk0,
          x_ag_tilde=xagt0,
          phi_ag_tilde=phit0,
          accepted=ok_c0,
      )

      def _cvx_cond(c: _UPFAGCvxCarry) -> jax.Array:
        return jnp.logical_and(
            jnp.logical_not(c.accepted),
            c.itr < self.max_backtracks,
        )

      def _cvx_body(c: _UPFAGCvxCarry) -> _UPFAGCvxCarry:
        nxt = c.itr + 1
        lk, ak, xmd, gmd, xkc, xagt, phit, ok = _trial_cvx(nxt)
        return _UPFAGCvxCarry(
            itr=nxt,
            lambda_k=lk,
            alpha_k=ak,
            x_md=xmd,
            grad_md=gmd,
            x_k_next=xkc,
            x_ag_tilde=xagt,
            phi_ag_tilde=phit,
            accepted=ok,
        )

      cvx_fin = _loop.while_loop(_cvx_cond, _cvx_body, cvx_init, jit=jit)
      sum_lam_next = st.sum_lambda + cvx_fin.lambda_k

      f_ag, grad_ag = oracle.smooth_value_and_grad(st.x_ag)
      phi_ag = f_ag + oracle.proximal_value(st.x_ag)
      if use_bb:

        def _bb_ncvx() -> tuple[jax.Array, jax.Array]:
          sk = tree.sub(st.x_ag, st.x_ag_old)
          _, g_old = oracle.smooth_value_and_grad(st.x_ag_old)
          yk = tree.sub(grad_ag, g_old)
          sty = tree.vdot(sk, yk)
          b_bb = jnp.where(
              sty > 0.0,
              jnp.maximum(
                  tree.norm_sq(sk) / jnp.maximum(sty, 1e-30), sigma_arr
              ),
              jnp.maximum(st.beta0, sigma_arr),
          )
          return b_bb, jnp.asarray(1, dtype=jnp.int32)

        beta_bb, bb_g2 = jax.lax.cond(
            st.outer_iters > 0,
            _bb_ncvx,
            lambda: (
                jnp.maximum(st.beta0, sigma_arr),
                jnp.asarray(0, dtype=jnp.int32),
            ),
        )
        extra_grads = extra_grads + bb_g2
      else:
        beta_bb = jnp.maximum(st.beta0, sigma_arr)

      k_plus_1 = (st.outer_iters + 1).astype(dtype)

      def _trial_ncvx(
          itr_i: jax.Array,
      ) -> tuple[
          jax.Array,
          _typing.PyTree,
          jax.Array,
          _typing.PyTree,
          jax.Array,
          jax.Array,
      ]:
        beta_k = beta_bb * (self.gamma_2 ** itr_i.astype(dtype))
        x_ag_bar = oracle.prox(
            tree.add_scaled(st.x_ag, -beta_k, grad_ag),
            beta_k,
        )
        f_bar, grad_bar = oracle.smooth_value_and_grad(x_ag_bar)
        phi_bar = f_bar + oracle.proximal_value(x_ag_bar)
        diff_bar = tree.sub(x_ag_bar, st.x_ag)
        rhs = (
            phi_ag
            - tree.norm_sq(diff_bar) / jnp.maximum(2.0 * beta_k, 1e-30)
            + self.gamma_3 / k_plus_1
            + 10.0 * eps_val * (1.0 + jnp.abs(phi_ag))
        )
        return beta_k, x_ag_bar, f_bar, grad_bar, phi_bar, phi_bar <= rhs

      b0, xbar0, fbar0, gbar0, phibar0, ok_n0 = _trial_ncvx(it0)
      ncvx_init = _UPFAGNocvxCarry(
          itr=it0,
          beta_k=b0,
          x_ag_bar=xbar0,
          f_ag_bar=fbar0,
          grad_ag_bar=gbar0,
          phi_ag_bar=phibar0,
          accepted=ok_n0,
      )

      def _ncvx_cond(c: _UPFAGNocvxCarry) -> jax.Array:
        return jnp.logical_and(
            jnp.logical_not(c.accepted),
            c.itr < self.max_backtracks,
        )

      def _ncvx_body(c: _UPFAGNocvxCarry) -> _UPFAGNocvxCarry:
        nxt = c.itr + 1
        bk, xb, fb, gb, phib, ok = _trial_ncvx(nxt)
        return _UPFAGNocvxCarry(
            itr=nxt,
            beta_k=bk,
            x_ag_bar=xb,
            f_ag_bar=fb,
            grad_ag_bar=gb,
            phi_ag_bar=phib,
            accepted=ok,
        )

      ncvx_fin = _loop.while_loop(_ncvx_cond, _ncvx_body, ncvx_init, jit=jit)

      # Exact subgradient at x_ag_bar:
      # (x_ag - x_ag_bar) / beta_k - grad_f(x_ag) in dh(x_ag_bar).
      v_bar = tree.add(
          tree.scale(
              1.0 / jnp.maximum(ncvx_fin.beta_k, 1e-30),
              tree.sub(st.x_ag, ncvx_fin.x_ag_bar),
          ),
          tree.sub(ncvx_fin.grad_ag_bar, grad_ag),
      )
      res_bar = tree.norm(v_bar, kind=self.norm_kind)

      # Eq. (2.11): best among {x_ag, x_ag_bar, x_ag_tilde}.
      use_bar = ncvx_fin.phi_ag_bar <= phi_ag
      cand_ag = jax.tree.map(
          lambda xb, xa: jnp.where(use_bar, xb, xa),
          ncvx_fin.x_ag_bar,
          st.x_ag,
      )
      cand_phi = jnp.where(use_bar, ncvx_fin.phi_ag_bar, phi_ag)
      use_tilde = cvx_fin.phi_ag_tilde < cand_phi
      x_ag_next = jax.tree.map(
          lambda xt, ca: jnp.where(use_tilde, xt, ca),
          cvx_fin.x_ag_tilde,
          cand_ag,
      )

      cand_finite = jnp.logical_and(
          tree.all_finite(ncvx_fin.x_ag_bar),
          jnp.isfinite(res_bar),
      )
      cvx_trials = cvx_fin.itr + 1
      ncvx_trials = ncvx_fin.itr + 1
      ls_ok = jnp.logical_and(cvx_fin.accepted, ncvx_fin.accepted)
      tracker = st.tracker.update_incumbent(
          ncvx_fin.x_ag_bar,
          v_bar,
          res_bar,
          ncvx_fin.phi_ag_bar,
      ).advance(
          inner_inc=cvx_trials + ncvx_trials,
          smooth_inc=2 * cvx_trials + ncvx_trials + 1 + extra_grads,
          grad_inc=cvx_trials + ncvx_trials + 1 + extra_grads,
          prox_inc=cvx_trials + ncvx_trials,
          max_iters=self.max_iters,
          line_search_failed=jnp.logical_not(ls_ok),
          finite_ok=cand_finite,
      )

      return UPFAGState(
          x_k=cvx_fin.x_k_next,
          x_ag=x_ag_next,
          x_ag_old=st.x_ag,
          x_md_prev=cvx_fin.x_md,
          x_ag_tilde_prev=cvx_fin.x_ag_tilde,
          sum_lambda=sum_lam_next,
          lambda0=st.lambda0,
          beta0=st.beta0,
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
      callback: _typing.CallbackFn[UPFAGState] | None = None,
      time_limit: float | None = None,
  ) -> base.Solution:
    """Solves an unconstrained `CompositeProblem` using `UPFAG`."""
    if problem.is_constrained:
      raise ValueError(
          "UPFAG solves unconstrained composite problems; pass `QPAIPP`, "
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
