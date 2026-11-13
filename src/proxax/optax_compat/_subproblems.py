"""Shared subproblem carries, mixins, and helpers for Optax transforms."""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp

from proxax import _typing
from proxax import tree
from proxax.optax_compat._trackers import ConstrainedOptaxTracker
from proxax.optax_compat._trackers import OptaxTracker
from proxax.oracle import composite
from proxax.solvers import base as solver_base
from proxax.solvers.inner import acg
from proxax.solvers.inner import pf_acg


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ACGSubproblemCarry:
  """Composable inner `ACG` subproblem carry for `aipp` and `qp_aipp`."""

  x: _typing.PyTree
  y: _typing.PyTree
  u: _typing.PyTree
  eta: jax.Array
  a_sum: jax.Array
  sub_psi_x0: jax.Array
  lam: jax.Array
  lipschitz: jax.Array

  @classmethod
  def create(
      cls,
      center: _typing.PyTree,
      center_obj: jax.Array,
      *,
      lam: jax.Array,
      lipschitz: jax.Array,
  ) -> ACGSubproblemCarry:
    """Initializes or resets an `ACGSubproblemCarry` at `center`."""
    dtype = lipschitz.dtype
    return cls(
        x=center,
        y=center,
        u=tree.zeros_like(center),
        eta=jnp.asarray(0.0, dtype=dtype),
        a_sum=jnp.asarray(0.0, dtype=dtype),
        sub_psi_x0=center_obj,
        lam=lam,
        lipschitz=lipschitz,
    )

  def step(
      self,
      oracle: composite.CompositeOracleProtocol,
      *,
      center: _typing.PyTree,
      inner_iter: jax.Array,
      config: acg.ACGConfig,
      has_known_m: bool,
      lower_curvature: float | None,
      use_adaptive: bool,
  ) -> tuple[acg.ACGState, jax.Array, jax.Array]:
    """Executes one `ACG` inner step on the proximal subproblem at `center`."""
    dtype = self.lipschitz.dtype
    inv_lam = 1.0 / self.lam
    if has_known_m and not use_adaptive:
      m_arr = jnp.asarray(lower_curvature, dtype=dtype)
      smooth_reg = jnp.minimum(m_arr, inv_lam)
      prox_reg = jnp.maximum(jnp.asarray(0.0, dtype=dtype), inv_lam - m_arr)
    else:
      smooth_reg = 0.5 * inv_lam
      prox_reg = 0.5 * inv_lam

    sub_oracle = composite.ProxRegularizedOracle(
        base=oracle,
        center=center,
        smooth_reg=smooth_reg,
        prox_reg=prox_reg,
    )
    acg_in = acg.ACGState(
        x=self.x,
        y=self.y,
        x0=center,
        psi_x0=self.sub_psi_x0,
        psi_y=self.sub_psi_x0,
        u=self.u,
        eta=self.eta,
        a_sum=self.a_sum,
        lipschitz=self.lipschitz + smooth_reg,
        iteration=inner_iter,
        status=jnp.asarray(int(acg.ACGStatus.RUNNING), dtype=jnp.int32),
        counts=solver_base.OracleCounts.zeros(),
    )
    acg_out = acg.step_acg(
        sub_oracle,
        acg_in,
        prox_center=center,
        lam=self.lam,
        config=config,
        check_convexity=use_adaptive,
        jit=True,
    )
    return acg_out, smooth_reg, inv_lam

  def continue_from(
      self,
      acg_out: acg.ACGState,
      *,
      smooth_reg: jax.Array,
  ) -> ACGSubproblemCarry:
    """Advances carry fields after a continuing inner `ACG` step."""
    est_L = jnp.maximum(
        acg_out.lipschitz - smooth_reg,
        0.5 * self.lipschitz,
    )
    return dataclasses.replace(
        self,
        x=acg_out.x,
        y=acg_out.y,
        u=acg_out.u,
        eta=acg_out.eta,
        a_sum=acg_out.a_sum,
        lipschitz=est_L,
    )


def _sub_lip_init(
    m_val: jax.Array,
    M_val: jax.Array,
    *,
    decay_curvature: bool,
    beta: float,
) -> jax.Array:
  """Computes initial rescaled subproblem smoothness `L_0` for `PF.ACG`."""
  dtype = M_val.dtype
  if decay_curvature:
    M_trial = jnp.maximum(
        jnp.asarray(1e-8, dtype=dtype),
        M_val / (1.0 + 0.5 * beta),
    )
    return M_trial / (2.0 * m_val) + 1.0
  return jnp.maximum(
      jnp.asarray(1.0, dtype=dtype),
      M_val / (2.0 * m_val) + 1.0,
  )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class PFAPDSubproblemCarry:
  """Composable inner `PF.ACG` subproblem carry for `apd`, `qp_apd`, `ialm`."""

  x: _typing.PyTree
  y: _typing.PyTree
  r: _typing.PyTree
  a_sum: jax.Array
  psi_y0: jax.Array
  psi_y: jax.Array
  lipschitz: jax.Array
  lower_curvature: jax.Array
  upper_curvature: jax.Array
  never_increased_m: jax.Array

  @classmethod
  def create(
      cls,
      center: _typing.PyTree,
      center_obj: jax.Array,
      *,
      lower_curvature: jax.Array,
      upper_curvature: jax.Array,
      never_increased_m: jax.Array,
      decay_curvature: bool,
      beta: float,
  ) -> PFAPDSubproblemCarry:
    """Initializes or resets a `PFAPDSubproblemCarry` at `center`."""
    dtype = upper_curvature.dtype
    psi0 = center_obj / (2.0 * lower_curvature)
    return cls(
        x=center,
        y=center,
        r=tree.zeros_like(center),
        a_sum=jnp.asarray(0.0, dtype=dtype),
        psi_y0=psi0,
        psi_y=psi0,
        lipschitz=_sub_lip_init(
            lower_curvature,
            upper_curvature,
            decay_curvature=decay_curvature,
            beta=beta,
        ),
        lower_curvature=lower_curvature,
        upper_curvature=upper_curvature,
        never_increased_m=never_increased_m,
    )

  def step(
      self,
      oracle: composite.CompositeOracleProtocol,
      *,
      center: _typing.PyTree,
      inner_iter: jax.Array,
      config: pf_acg.PFACGConfig,
      opt_tol: jax.Array,
      norm_kind: str,
  ) -> tuple[pf_acg.PFACGState, _typing.PyTree, _typing.PyTree, jax.Array]:
    """Executes one `PF.ACG` step and forms refinement-free certificate `v`."""
    two_m = 2.0 * self.lower_curvature
    sub_oracle = composite.RescaledProxOracle(
        base=oracle,
        center=center,
        scale=1.0 / two_m,
    )
    running_code = jnp.asarray(int(pf_acg.PFACGStatus.RUNNING), dtype=jnp.int32)
    pf_in = pf_acg.PFACGState(
        x=self.x,
        y=self.y,
        y0=center,
        psi_y0=self.psi_y0,
        psi_y=self.psi_y,
        r=self.r,
        a_sum=self.a_sum,
        lipschitz=self.lipschitz,
        iteration=inner_iter,
        status=running_code,
        counts=solver_base.OracleCounts.zeros(),
    )
    pf_out = pf_acg.step_pf_acg(
        sub_oracle,
        pf_in,
        config=config,
        two_m=two_m,
        opt_tol=opt_tol,
        norm_kind=norm_kind,
        jit=True,
    )
    u_cand = tree.scale(two_m, pf_out.r)
    v_cand = tree.add(u_cand, tree.scale(two_m, tree.sub(center, pf_out.y)))
    cand_opt = tree.norm(v_cand, kind=norm_kind)
    return pf_out, u_cand, v_cand, cand_opt

  def continue_from(self, pf_out: pf_acg.PFACGState) -> PFAPDSubproblemCarry:
    """Advances carry fields after a continuing inner `PF.ACG` step."""
    return dataclasses.replace(
        self,
        x=pf_out.x,
        y=pf_out.y,
        r=pf_out.r,
        a_sum=pf_out.a_sum,
        psi_y=pf_out.psi_y,
        lipschitz=pf_out.lipschitz,
    )


class OptaxTrackerMixin:
  """Zero-slot `@property` forwarding mixin for states composing `tracker`."""

  __slots__ = ()
  tracker: OptaxTracker | ConstrainedOptaxTracker

  @property
  def initialized(self) -> jax.Array:
    """Whether step-0 certificate bootstrap has completed."""
    return self.tracker.initialized

  @property
  def converged(self) -> jax.Array:
    """Whether target stopping tolerances are satisfied."""
    return self.tracker.converged

  @property
  def outer_iter(self) -> jax.Array:
    """Completed outer proximal iterations."""
    return self.tracker.outer_iter

  @property
  def inner_iter(self) -> jax.Array:
    """Current inner subproblem iteration."""
    return self.tracker.inner_iter

  @property
  def total_steps(self) -> jax.Array:
    """Cumulative `tx.update` calls."""
    return self.tracker.total_steps

  @property
  def center(self) -> _typing.PyTree:
    """Active proximal center `x_k^-`."""
    return self.tracker.center

  @property
  def center_obj(self) -> jax.Array:
    """Objective value at `center`."""
    return self.tracker.center_obj

  @property
  def best_x(self) -> _typing.PyTree:
    """Best certified primal iterate `x_hat`."""
    return self.tracker.best_x

  @property
  def subgrad(self) -> _typing.PyTree:
    """Certified subgradient `v_hat` at `best_x`."""
    return self.tracker.subgrad

  @property
  def opt_residual(self) -> jax.Array:
    """Stationarity residual norm `||v_hat||` at `best_x`."""
    return self.tracker.opt_residual


class ConstrainedOptaxTrackerMixin(OptaxTrackerMixin):
  """Zero-slot `@property` forwarding mixin for constrained Optax states."""

  __slots__ = ()
  tracker: ConstrainedOptaxTracker

  @property
  def stage_iter(self) -> jax.Array:
    """Completed outer penalty or Augmented Lagrangian stages."""
    return self.tracker.stage_iter

  @property
  def stage_outer_iter(self) -> jax.Array:
    """Proximal outer iterations completed in the active stage."""
    return self.tracker.stage_outer_iter

  @property
  def penalty(self) -> jax.Array:
    """Active quadratic penalty parameter `c_k`."""
    return self.tracker.penalty

  @property
  def dual(self) -> _typing.PyTree:
    """Certified dual multiplier `p_hat` at `best_x`."""
    return self.tracker.dual

  @property
  def feas_residual(self) -> jax.Array:
    """Primal feasibility / complementarity residual at `best_x`."""
    return self.tracker.feas_residual


class ACGSubproblemMixin:
  """Zero-slot `@property` forwarding mixin for `ACGSubproblemCarry`."""

  __slots__ = ()
  subproblem: ACGSubproblemCarry

  @property
  def acg_x(self) -> _typing.PyTree:
    """Inner `ACG` auxiliary iterate `x`."""
    return self.subproblem.x

  @property
  def acg_y(self) -> _typing.PyTree:
    """Inner `ACG` primal iterate `y`."""
    return self.subproblem.y

  @property
  def acg_u(self) -> _typing.PyTree:
    """Inner `ACG` approximate subgradient `u`."""
    return self.subproblem.u

  @property
  def acg_eta(self) -> jax.Array:
    """Inner `ACG` affine gap `eta`."""
    return self.subproblem.eta

  @property
  def acg_a_sum(self) -> jax.Array:
    """Inner `ACG` cumulative momentum weight `A_j`."""
    return self.subproblem.a_sum

  @property
  def sub_psi_x0(self) -> jax.Array:
    """Proximal subproblem objective at `center`."""
    return self.subproblem.sub_psi_x0

  @property
  def lam(self) -> jax.Array:
    """Active proximal step size `lambda_k`."""
    return self.subproblem.lam

  @property
  def lipschitz(self) -> jax.Array:
    """Estimated smoothness constant `L_k`."""
    return self.subproblem.lipschitz


class PFAPDSubproblemMixin:
  """Zero-slot `@property` forwarding mixin for `PFAPDSubproblemCarry`."""

  __slots__ = ()
  subproblem: PFAPDSubproblemCarry

  @property
  def pf_x(self) -> _typing.PyTree:
    """Inner `PF.ACG` auxiliary iterate `x`."""
    return self.subproblem.x

  @property
  def pf_y(self) -> _typing.PyTree:
    """Inner `PF.ACG` primal iterate `y`."""
    return self.subproblem.y

  @property
  def pf_r(self) -> _typing.PyTree:
    """Inner `PF.ACG` rescaled residual `r`."""
    return self.subproblem.r

  @property
  def pf_a_sum(self) -> jax.Array:
    """Inner `PF.ACG` cumulative momentum weight `A_j`."""
    return self.subproblem.a_sum

  @property
  def pf_psi_y0(self) -> jax.Array:
    """Rescaled subproblem objective at `center`."""
    return self.subproblem.psi_y0

  @property
  def pf_psi_y(self) -> jax.Array:
    """Rescaled subproblem objective at `pf_y`."""
    return self.subproblem.psi_y

  @property
  def pf_lipschitz(self) -> jax.Array:
    """Rescaled inner `PF.ACG` smoothness estimate."""
    return self.subproblem.lipschitz

  @property
  def lower_curvature(self) -> jax.Array:
    """Active lower curvature estimate `m_hat`."""
    return self.subproblem.lower_curvature

  @property
  def upper_curvature(self) -> jax.Array:
    """Active upper smoothness estimate `M_hat`."""
    return self.subproblem.upper_curvature

  @property
  def never_increased_m(self) -> jax.Array:
    """Whether `lower_curvature` has never been multiplied by `alpha`."""
    return self.subproblem.never_increased_m


# ---------------------------------------------------------------------------
# Shared calibration & acceptance helpers
# ---------------------------------------------------------------------------


def _initial_lam(
    *,
    lam_init: float | None,
    lower_curvature: float | None,
    lipschitz_init: float,
    dtype: jax.typing.DTypeLike,
) -> jax.Array:
  """Computes initial proximal step size `lambda_0` before step-0 bootstrap."""
  if lam_init is not None:
    return jnp.asarray(lam_init, dtype=dtype)
  if lower_curvature is not None:
    return jnp.asarray(
        0.5 / max(lower_curvature, 1e-12)
        if lower_curvature > 0.0
        else 10.0 / max(lipschitz_init, 1e-6),
        dtype=dtype,
    )
  return jnp.asarray(2.0 / max(lipschitz_init, 1e-6), dtype=dtype)


def _check_aipp_rejection(
    acg_out: acg.ACGState,
    *,
    center: _typing.PyTree,
    center_obj: jax.Array,
    phi_y: jax.Array,
    lam: jax.Array,
    inv_lam: jax.Array,
    sigma: float,
    use_adaptive: bool,
) -> jax.Array:
  """Evaluates whether an adaptive `AIPP` outer step should be rejected."""
  dtype = center_obj.dtype
  rhs_vec = tree.add(
      tree.scale(inv_lam, tree.sub(center, acg_out.y)),
      acg_out.u,
  )
  req_desc = 0.5 * lam * (1.0 - sigma**2) * tree.norm_sq(rhs_vec)
  slack = 100.0 * tree.eps(dtype) * (1.0 + jnp.abs(center_obj))
  desc_ok = (center_obj - phi_y + slack) >= req_desc
  noncvx_code = jnp.asarray(
      int(acg.ACGStatus.NONCONVEX_DETECTED),
      dtype=jnp.int32,
  )
  return jnp.logical_and(
      jnp.asarray(use_adaptive),
      jnp.logical_or(acg_out.status == noncvx_code, jnp.logical_not(desc_ok)),
  )


def _validate_apd_params(
    *,
    alpha: float,
    beta: float,
    theta: float,
    rho: float | None,
    decay_curvature: bool,
    max_inner_iters: int,
) -> tuple[float, pf_acg.PFACGConfig]:
  """Validates `APD` hyperparameters and builds `PFACGConfig`."""
  if alpha <= 1.0:
    raise ValueError("`alpha` must be strictly greater than 1.")
  if beta <= 1.0:
    raise ValueError("`beta` must be strictly greater than 1.")
  if theta <= 2.0:
    raise ValueError("`theta` must be strictly greater than 2.")
  if rho is not None and not (0.0 < rho < 1.0):
    raise ValueError("`rho` must lie in `(0, 1)`.")
  eff_rho = (
      rho
      if rho is not None
      else (
          1.0 / math.sqrt(alpha)
          if decay_curvature
          else 1.0 / math.sqrt(2.0 * alpha)
      )
  )
  pf_cfg = pf_acg.PFACGConfig(
      mu=0.5,
      sigma=eff_rho / math.sqrt(2.0),
      theta=theta,
      beta=beta,
      max_iters=max_inner_iters,
  )
  return eff_rho, pf_cfg


def _initial_lower_curvature(
    *,
    lower_curvature_init: float | None,
    lower_curvature: float | None,
    opt_tol: float,
    lower_curvature_min: float,
    lower_curvature_max: float,
    dtype: jax.typing.DTypeLike,
) -> jax.Array:
  """Calibrates initial weak-convexity estimate `m_0` for Optax `APD`."""
  if lower_curvature_init is not None:
    m0_f = lower_curvature_init
  elif lower_curvature is not None:
    m0_f = (
        lower_curvature
        if lower_curvature > 0.0
        else max(opt_tol, lower_curvature_min)
    )
  else:
    m0_f = 1.0
  return jnp.clip(
      jnp.asarray(m0_f, dtype=dtype),
      jnp.asarray(lower_curvature_min, dtype=dtype),
      jnp.asarray(lower_curvature_max, dtype=dtype),
  )


def _evaluate_apd_subproblem(
    pf_out: pf_acg.PFACGState,
    *,
    center: _typing.PyTree,
    center_obj: jax.Array,
    phi_y: jax.Array,
    u_cand: _typing.PyTree,
    v_cand: _typing.PyTree,
    m_hat: jax.Array,
    theta: float,
    eff_rho: float,
    done_flag: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Returns `(rejected, dist_sq, M_from_acg)` for finished `PF.ACG` run."""
  dtype = center_obj.dtype
  eps_val = tree.eps(dtype)
  eps_sq = eps_val**2
  phi_slack = 100.0 * eps_val * (1.0 + jnp.abs(center_obj))
  desc_ok = (
      tree.norm_sq(v_cand)
      <= 2.0 * theta * m_hat * (center_obj - phi_y + phi_slack) + eps_sq
  )
  dist_sq = tree.norm_sq(tree.sub(pf_out.y, center))
  prox_err_ok = (
      tree.norm_sq(u_cand)
      <= 2.0 * ((eff_rho * m_hat) ** 2) * dist_sq + eps_sq
  )
  bad_code = jnp.asarray(int(pf_acg.PFACGStatus.BAD_TERMINATION), jnp.int32)
  ls_fail_code = jnp.asarray(
      int(pf_acg.PFACGStatus.LINE_SEARCH_FAILED), jnp.int32
  )
  rejected = jnp.logical_and(
      jnp.logical_not(done_flag),
      jnp.logical_and(
          pf_out.status != ls_fail_code,
          jnp.logical_or(
              pf_out.status == bad_code,
              jnp.logical_not(jnp.logical_and(desc_ok, prox_err_ok)),
          ),
      ),
  )
  M_from_acg = jnp.maximum(
      jnp.asarray(1e-8, dtype=dtype),
      2.0 * m_hat * (pf_out.lipschitz - 1.0),
  )
  return rejected, dist_sq, M_from_acg


def _accepted_apd_curvatures(
    *,
    m_hat: jax.Array,
    upper_curvature: jax.Array,
    never_increased_m: jax.Array,
    M_from_acg: jax.Array,
    dist_sq: jax.Array,
    phi_y: jax.Array,
    alpha: float,
    decay_curvature: bool,
    m_min_arr: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Computes `(m_next, M_est)` after an accepted `APD` proximal step."""
  dtype = phi_y.dtype
  eps_val = tree.eps(dtype)
  eps_sq = eps_val**2
  quad_curv = 0.5 * M_from_acg * dist_sq
  noise_floor = 100.0 * eps_val * (1.0 + jnp.abs(phi_y))
  M_est = jnp.where(
      quad_curv > noise_floor,
      M_from_acg,
      jnp.maximum(M_from_acg, upper_curvature),
  )
  if decay_curvature:
    m_relaxed = jnp.maximum(m_min_arr, m_hat / (1.0 + 0.5 * alpha))
    m_next = jnp.where(dist_sq > eps_sq, m_relaxed, m_hat)
  else:
    m_bitonic = jnp.maximum(m_min_arr, m_hat / alpha)
    m_next = jnp.where(never_increased_m, m_bitonic, m_hat)
  return m_next, M_est


# ---------------------------------------------------------------------------
# Public Optax transformations and state classes
# ---------------------------------------------------------------------------

