"""Inner-Accelerated Inexact Composite Gradient (`IA-ICG`) spectral solver."""

from __future__ import annotations

import dataclasses
import typing

import jax
import jax.numpy as jnp

from proxax import _typing
from proxax import problem as problem_mod
from proxax import tree
from proxax.oracle import spectral
from proxax.solvers import base
from proxax.solvers.inner import acg
from proxax.solvers.unconstrained import _icg_common


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IAICGState(base.SolverTrackerMixin):
  """Outer iteration state of the `IAICG` (`IA-ICG`) spectral solver."""

  z0: jax.Array
  z0_svals: jax.Array
  lam: jax.Array
  xi: jax.Array
  m2_est: jax.Array
  tracker: base.SolverTracker


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IAICG:
  """Inner-Accelerated Inexact Composite Gradient (`IA-ICG`) spectral solver.

  Implements `src/solvers/IA_ICG.m` from `wwkong/nc_opt` based on W. Kong and
  R. D. C. Monteiro, *"Accelerated inexact composite gradient methods for
  nonconvex spectral optimization problems,"* arXiv:2007.11772, 2020
  (Comput. Optim. Appl., 2023).

  Minimizes `f_1(X) + (f_2^V o sigma)(X) + (h^V o sigma)(X)` by redistributing
  curvature `xi` from `f_1` into `f_2^V`, computing a single thin SVD per outer
  step at `Z_0 - lam * grad f_1_tilde(Z_0)`, solving the reduced `r`-dimensional
  singular-value subproblem via `ACG`, and refining an exact matrix certificate
  `V_hat in grad f_1(Z_hat) + grad f_2(Z_hat) + dh(Z_hat)`.

  Attributes:
    opt_tol: Absolute stationarity tolerance `rho > 0`.
    rel_opt_tol: Optional relative stationarity tolerance vs step-0 norm.
    norm_kind: Residual norm (`'l2'` or `'linf'`).
    step_type: `'adaptive'` (balances `v1_hat` vs `v2_hat`) or `'constant'`.
    xi_init: Optional initial redistributed curvature `xi_0 > 0`.
    lam_init: Optional initial outer step size `lambda_0 > 0`.
    sigma: Optional inner `ACG` tolerance in `(0, 1)` (defaults to `sqrt(0.9)`).
    delta_tol: Relative descent/refinement tolerance (`1e-2`).
    lipschitz_init: Seed smoothness used when `M_1` is unknown.
    lam_min: Minimum allowed step size `lambda`.
    lam_max: Maximum allowed step size `lambda`.
    max_outer_iters: Maximum outer `IA-ICG` iterations.
    max_inner_iters: Maximum 1D `ACG` iterations per outer step.
  """

  opt_tol: float = 1e-5
  rel_opt_tol: float = 0.0
  norm_kind: str = "l2"
  step_type: str = "adaptive"
  xi_init: float | None = None
  lam_init: float | None = None
  sigma: float | None = None
  delta_tol: float = 1e-2
  lipschitz_init: float = 1.0
  lam_min: float = 1e-10
  lam_max: float = 1e8
  max_outer_iters: int = 500
  max_inner_iters: int = 1000

  def __post_init__(self) -> None:
    """Validates `IAICG` hyperparameters."""
    if self.step_type not in ("adaptive", "constant"):
      raise ValueError(
          "`step_type` must be 'adaptive' or 'constant'; "
          f"got {self.step_type!r}."
      )
    if self.sigma is not None and not 0.0 < self.sigma < 1.0:
      raise ValueError("`sigma` must lie strictly in (0, 1).")

  def solve_oracle(
      self,
      oracle: spectral.SpectralOracle,
      x0: jax.Array,
      *,
      opt_tol_override: _typing.Scalar | None = None,
      jit: bool = True,
      callback: _typing.CallbackFn[IAICGState] | None = None,
      time_limit: float | None = None,
  ) -> IAICGState:
    """Runs `IAICG` on a `SpectralOracle` starting from 2D matrix `x0`."""
    setup = _icg_common.init_icg_setup(
        oracle,
        x0,
        solver_name="IAICG",
        opt_tol=self.opt_tol,
        rel_opt_tol=self.rel_opt_tol,
        opt_tol_override=opt_tol_override,
        norm_kind=self.norm_kind,
        lipschitz_init=self.lipschitz_init,
        xi_init=self.xi_init,
        lam_init=self.lam_init,
        sigma=self.sigma,
        max_inner_iters=self.max_inner_iters,
        jit=jit,
    )
    dtype = setup.dtype
    m2_low = setup.m2_low
    acg_cfg = setup.acg_cfg

    init_state = IAICGState(
        z0=setup.z0_hat,
        z0_svals=setup.z0_svals,
        lam=setup.lam0,
        xi=setup.xi0,
        m2_est=setup.m2_est0 + setup.xi0,
        tracker=setup.tracker0,
    )

    def _body(st: IAICGState) -> IAICGState:
      eps_val = tree.eps(dtype)
      f1_z0, g1_z0 = oracle.f1_redistributed_vag(st.z0, st.xi)
      w_mat = st.z0 - st.lam * g1_z0
      p_mat, x0_vec, qt_mat = jnp.linalg.svd(w_mat, full_matrices=False)

      vec_oracle = oracle.vector_subproblem_oracle(st.lam, x0_vec, st.xi)
      d0_vec = spectral.diag_pt_mat_q(p_mat, st.z0, qt_mat)
      orth_gap_sq = jnp.maximum(
          jnp.asarray(0.0, dtype=dtype),
          jnp.sum(st.z0 * st.z0) - jnp.sum(d0_vec * d0_vec),
      )
      acg_cfg_step = dataclasses.replace(acg_cfg, rhs_offset_sq=orth_gap_sq)
      sub_L_init = st.lam * st.m2_est + 0.5
      acg_st = acg.run_acg(
          vec_oracle,
          st.z0_svals,
          prox_center=d0_vec,
          lam=jnp.asarray(1.0, dtype=dtype),
          lipschitz_init=sub_L_init,
          config=acg_cfg_step,
          check_convexity=True,
          jit=jit,
      )

      z_vec = typing.cast(jax.Array, acg_st.y)
      v_vec = typing.cast(jax.Array, acg_st.u)
      z_mat = spectral.reconstruct(p_mat, z_vec, qt_mat)
      v_mat = spectral.reconstruct(p_mat, v_vec, qt_mat)

      noncvx_code = jnp.asarray(
          int(acg.ACGStatus.NONCONVEX_DETECTED), dtype=jnp.int32
      )
      ls_fail_code = jnp.asarray(
          int(acg.ACGStatus.LINE_SEARCH_FAILED), dtype=jnp.int32
      )
      acg_bad = jnp.logical_or(
          acg_st.status == noncvx_code,
          acg_st.status == ls_fail_code,
      )

      if oracle.f1_obj.curvature_upper is None:
        f1_z = oracle.f1_redistributed_val(z_mat, st.xi)
        dz_mat = z_mat - st.z0
        f1_slack = 100.0 * eps_val * (1.0 + jnp.abs(f1_z0))
        f1_bad = f1_z > f1_z0 + jnp.sum(g1_z0 * dz_mat) + f1_slack
        extra_f1 = jnp.asarray(1, dtype=jnp.int32)
      else:
        f1_bad = jnp.asarray(False)
        extra_f1 = jnp.asarray(0, dtype=jnp.int32)

      d_z0, p_s0, p_y0 = spectral.delta_mu(
          oracle,
          lam=st.lam,
          xi=st.xi,
          mu=0.5,
          y0_mat=st.z0,
          f1_y0=f1_z0,
          g1_y0=g1_z0,
          y_mat=z_mat,
          y_vec=z_vec,
          s_mat=st.z0,
          s_vec=st.z0_svals,
          v_mat=v_mat,
      )
      base0 = jnp.maximum(
          jnp.maximum(jnp.abs(p_s0), jnp.abs(p_y0)),
          jnp.asarray(self.delta_tol, dtype=dtype),
      )
      desc_bad = (d_z0 - acg_st.eta) / base0 >= self.delta_tol

      ref = spectral.refine_icg(
          oracle,
          p_mat=p_mat,
          qt_mat=qt_mat,
          z0_mat=st.z0,
          g1_z0=g1_z0,
          x0_vec=x0_vec,
          z_vec=z_vec,
          v_vec=v_vec,
          lam=st.lam,
          xi=st.xi,
          m2_est=st.m2_est,
          jit=jit,
      )

      d_zh, p_sh, p_yh = spectral.delta_mu(
          oracle,
          lam=st.lam,
          xi=st.xi,
          mu=0.5,
          y0_mat=st.z0,
          f1_y0=f1_z0,
          g1_y0=g1_z0,
          y_mat=z_mat,
          y_vec=z_vec,
          s_mat=ref.z_hat_mat,
          s_vec=jnp.sort(jnp.abs(ref.z_hat_vec))[::-1],
          v_mat=v_mat,
      )
      baseh = jnp.maximum(
          jnp.maximum(jnp.abs(p_sh), jnp.abs(p_yh)),
          jnp.asarray(self.delta_tol, dtype=dtype),
      )
      ref_bad = (d_zh - acg_st.eta) / baseh >= self.delta_tol

      res_cand = tree.norm(ref.v_hat_mat, kind=self.norm_kind)
      cand_finite = jnp.logical_and(
          tree.all_finite(ref.z_hat_mat),
          jnp.isfinite(res_cand),
      )

      rejected = jnp.logical_or(
          jnp.logical_or(acg_bad, f1_bad),
          jnp.logical_or(desc_bad, ref_bad),
      )

      xi_max = jnp.asarray(1e6, dtype=dtype)
      xi_next = jnp.where(rejected, jnp.minimum(2.0 * st.xi, xi_max), st.xi)
      m2_base_cur = jnp.maximum(ref.m2_used - st.xi, 0.0)
      m2_est_next = m2_base_cur + xi_next

      if self.step_type == "adaptive":
        scale_z = (
            st.lam * (m2_base_cur + 2.0 * jnp.maximum(m2_low, 0.0)) + 1.0
        ) / st.lam
        v1_hat = ref.q_hat_mat - scale_z * (z_mat - ref.z_hat_mat)
        n_v1 = tree.norm(v1_hat, kind=self.norm_kind)
        n_v2 = tree.norm(ref.v_hat_mat - v1_hat, kind=self.norm_kind)
        ratio = n_v1 / jnp.maximum(n_v2, 1e-30)
        lam_adap = jnp.where(
            ratio < 0.5,
            st.lam * jnp.sqrt(0.5),
            jnp.where(ratio > 2.0, st.lam * jnp.sqrt(2.0), st.lam),
        )
        lam_acc = jnp.clip(
            lam_adap,
            jnp.asarray(self.lam_min, dtype=dtype),
            jnp.asarray(self.lam_max, dtype=dtype),
        )
      else:
        lam_acc = st.lam

      lam_rej = jnp.where(
          f1_bad,
          jnp.maximum(0.5 * st.lam, jnp.asarray(self.lam_min, dtype=dtype)),
          st.lam,
      )
      lam_next = jnp.where(rejected, lam_rej, lam_acc)
      z0_next = jnp.where(rejected, st.z0, z_mat)
      z_svals_sorted = jnp.sort(jnp.abs(z_vec))[::-1]
      z0_svals_next = jnp.where(rejected, st.z0_svals, z_svals_sorted)

      tracker = st.tracker.update_incumbent(
          ref.z_hat_mat,
          ref.v_hat_mat,
          res_cand,
          ref.obj_val,
      ).advance(
          inner_inc=acg_st.iteration,
          smooth_inc=1 + extra_f1 + ref.f1_evals,
          grad_inc=1 + ref.f1_grad_evals,
          prox_inc=1,
          counts_inc=acg_st.counts,
          max_iters=self.max_outer_iters,
          finite_ok=cand_finite,
      )

      return IAICGState(
          z0=z0_next,
          z0_svals=z0_svals_next,
          lam=lam_next,
          xi=xi_next,
          m2_est=m2_est_next,
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
      problem: spectral.SpectralProblem | problem_mod.CompositeProblem,
      x0: _typing.PyTree,
      *,
      jit: bool = True,
      callback: _typing.CallbackFn[IAICGState] | None = None,
      time_limit: float | None = None,
  ) -> base.Solution:
    """Solves a `SpectralProblem` or nuclear `CompositeProblem` via `IAICG`."""
    spec_oracle = spectral.as_spectral_oracle(problem)
    x0_arr = typing.cast(jax.Array, x0)
    fin = self.solve_oracle(
        spec_oracle,
        x0_arr,
        jit=jit,
        callback=callback,
        time_limit=time_limit,
    )
    return fin.to_solution()
