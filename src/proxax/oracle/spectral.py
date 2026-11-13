"""Spectral composite oracles and problems (Kong & Monteiro 2020)."""

from __future__ import annotations

import dataclasses
import typing

import jax
import jax.numpy as jnp

from proxax import _loop
from proxax import _typing
from proxax import tree
from proxax.oracle import terms
from proxax.prox import base as prox_base

if typing.TYPE_CHECKING:
  from proxax.oracle.spectral_problem import SpectralProblem as SpectralProblem
  from proxax.oracle.spectral_problem import SpectralProx as SpectralProx
  from proxax.oracle.spectral_problem import (
    as_spectral_oracle as as_spectral_oracle,
  )


def reconstruct(
    p_mat: jax.Array,
    s_vec: jax.Array,
    qt_mat: jax.Array,
) -> jax.Array:
  """Computes `P diag(s) Q^T` in full IEEE precision (bypassing TF32)."""
  return jnp.matmul(
      p_mat * s_vec[None, :],
      qt_mat,
      precision=jax.lax.Precision.HIGHEST,
  )


def diag_pt_mat_q(
    p_mat: jax.Array,
    mat: jax.Array,
    qt_mat: jax.Array,
) -> jax.Array:
  """Computes `diag(P^T mat Q)` in full IEEE precision (bypassing TF32)."""
  mat_q = jnp.matmul(mat, qt_mat.T, precision=jax.lax.Precision.HIGHEST)
  return jnp.sum(p_mat * mat_q, axis=0)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class VectorSpectralSubOracle:
  """1D singular-value proximal subproblem oracle for `IA-ICG` / `DA-ICG`.

  Represents `Psi_lam(x) = psi^s(x) + psi^n(x)` on `R^r`:

  - `psi^s(x) = lam * (f2(x) + 0.5 * xi * ||x||_2^2) + 0.25 * ||x - x0||_2^2`
  - `psi^n(x) = lam * prox_vec(x) + 0.25 * ||x - x0||_2^2`

  Splitting `0.5 * ||x - x0||_2^2` into `0.25 + 0.25` matches `mu = 1 / 2` in
  `IA_ICG.m` / `DA_ICG.m` while preserving exact analytical proximal steps.
  """

  f2_vec: terms.SmoothObjective
  prox_vec: prox_base.ProxOperator
  lam: jax.Array
  x_hat: jax.Array
  xi: jax.Array

  @property
  def curvature_upper(self) -> jax.Array | float | None:
    """Upper smoothness `lam * (M2 + xi) + 0.5` when `M2` is known."""
    if self.f2_vec.curvature_upper is None:
      return None
    return self.lam * (self.f2_vec.curvature_upper + self.xi) + 0.5

  @property
  def curvature_lower(self) -> jax.Array | float | None:
    """Weak-convexity bound of `psi^s` when `m2` is known."""
    if self.f2_vec.curvature_lower is None:
      return None
    rem = self.lam * (self.f2_vec.curvature_lower - self.xi) - 0.5
    return jnp.maximum(0.0, rem)

  @property
  def strong_convexity(self) -> jax.Array:
    """Strong convexity `0.5 + lam * prox_vec.strong_convexity` of `psi^n`."""
    return 0.5 + self.lam * jnp.asarray(
        self.prox_vec.strong_convexity,
        dtype=self.lam.dtype,
    )

  def smooth_value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates `psi^s(x)` on singular-value vector `x`."""
    x_arr = typing.cast(jax.Array, x)
    f2_val = self.f2_vec.value(x_arr) + 0.5 * self.xi * jnp.sum(x_arr * x_arr)
    diff = x_arr - self.x_hat
    return self.lam * f2_val + 0.25 * jnp.sum(diff * diff)

  def smooth_value_and_grad(
      self,
      x: _typing.PyTree,
  ) -> tuple[jax.Array, _typing.PyTree]:
    """Evaluates `(psi^s(x), grad psi^s(x))`."""
    x_arr = typing.cast(jax.Array, x)
    v2, g2_raw = self.f2_vec.value_and_grad(x_arr)
    g2 = typing.cast(jax.Array, g2_raw) + self.xi * x_arr
    f2_val = v2 + 0.5 * self.xi * jnp.sum(x_arr * x_arr)
    diff = x_arr - self.x_hat
    val = self.lam * f2_val + 0.25 * jnp.sum(diff * diff)
    grad = self.lam * g2 + 0.5 * diff
    return val, grad

  def proximal_value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates `psi^n(x)`."""
    x_arr = typing.cast(jax.Array, x)
    diff = x_arr - self.x_hat
    return self.lam * self.prox_vec.value(x_arr) + 0.25 * jnp.sum(diff * diff)

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates `Psi_lam(x) = psi^s(x) + psi^n(x)`."""
    return self.smooth_value(x) + self.proximal_value(x)

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Evaluates exact shifted `prox_{step_size * psi^n}(x)`."""
    x_arr = typing.cast(jax.Array, x)
    tau_half = 0.5 * step_size
    denom = 1.0 + tau_half
    shifted = (x_arr + tau_half * self.x_hat) / denom
    return self.prox_vec.prox(shifted, (step_size * self.lam) / denom)

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Projects `x` onto `cl(dom(prox_vec))`."""
    return self.prox_vec.domain_project(x)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class SpectralRefinementResult:
  """Certified matrix pair `(z_hat_mat, v_hat_mat)` produced by `refine_icg`."""

  z_hat_vec: jax.Array
  z_hat_mat: jax.Array
  q_hat_mat: jax.Array
  v_hat_mat: jax.Array
  m2_used: jax.Array
  obj_val: jax.Array
  f1_evals: jax.Array
  f1_grad_evals: jax.Array
  vec_evals: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _VecRefineCarry:
  """Line-search carry for unknown `M2` in `refine_icg`."""

  m2_plus: jax.Array
  z_hat: jax.Array
  f2_z_hat: jax.Array
  accepted: jax.Array
  trials: jax.Array


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class SpectralOracle:
  """Matrix spectral oracle `f1(X) + (f2_vec o sigma)(X) + (h o sigma)(X)`.

  Implements `src/oracles/SpectralOracle.m` from `wwkong/nc_opt` (Kong &
  Monteiro, *arXiv:2007.11772*, 2020) while satisfying `CompositeOracleProtocol`
  so general composite solvers (`AIPP`, `APD`, `ECG`, etc.) can also run on it.

  Attributes:
    f1: General smooth matrix term `f_1(X)` on `R^{m x n}`.
    f2_vec: Absolutely symmetric smooth singular-value term `f_2^V(sigma(X))`.
    prox_vec: Absolutely symmetric convex singular-value regularizer
      `h^V(sigma(X))` (e.g., `prox.L1(weight)` for nuclear norm).
    domain_proj: Optional matrix projection `Pi_Omega(X)` onto a closed convex
      superset `Omega` containing `dom(h)` (used by `DA-ICG`).
  """

  f1: terms.SmoothLike = dataclasses.field(default_factory=terms.ZeroSmooth)
  f2_vec: terms.SmoothLike = dataclasses.field(default_factory=terms.ZeroSmooth)
  prox_vec: prox_base.ProxOperator = dataclasses.field(
      default_factory=prox_base.Zero
  )
  domain_proj: typing.Callable[[jax.Array], jax.Array] | None = None

  def __post_init__(self) -> None:
    """Coerces callable `f1` and `f2_vec` fields into `SmoothObjective`."""
    if self.f1 is None or (
        callable(self.f1) and not isinstance(self.f1, terms.SmoothObjective)
    ):
      object.__setattr__(self, "f1", terms.as_smooth_term(self.f1))
    if self.f2_vec is None or (
        callable(self.f2_vec)
        and not isinstance(self.f2_vec, terms.SmoothObjective)
    ):
      object.__setattr__(self, "f2_vec", terms.as_smooth_term(self.f2_vec))

  @property
  def f1_obj(self) -> terms.SmoothObjective:
    """Normalized `SmoothObjective` for `f1`."""
    return typing.cast(terms.SmoothObjective, self.f1)

  @property
  def f2_vec_obj(self) -> terms.SmoothObjective:
    """Normalized `SmoothObjective` for `f2_vec`."""
    return typing.cast(terms.SmoothObjective, self.f2_vec)

  @property
  def curvature_upper(self) -> jax.Array | float | None:
    """Combined smoothness bound `M1 + M2` when both are known."""
    u1 = self.f1_obj.curvature_upper
    u2 = self.f2_vec_obj.curvature_upper
    if u1 is None or u2 is None:
      return None
    return u1 + u2

  @property
  def curvature_lower(self) -> jax.Array | float | None:
    """Combined weak-convexity bound `m1 + m2` when both are known."""
    l1 = self.f1_obj.curvature_lower
    l2 = self.f2_vec_obj.curvature_lower
    if l1 is None or l2 is None:
      return None
    return l1 + l2

  @property
  def strong_convexity(self) -> jax.Array | float:
    """Strong convexity constant `mu >= 0` of `prox_vec`."""
    return self.prox_vec.strong_convexity

  def smooth_value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates `f1(X) + f2_vec(sigma(X))`."""
    mat = typing.cast(jax.Array, x)
    v1 = self.f1_obj.value(mat)
    if isinstance(self.f2_vec_obj, terms.ZeroSmooth):
      return v1
    s_vec = jnp.linalg.svdvals(mat)
    return v1 + self.f2_vec_obj.value(s_vec)

  def smooth_value_and_grad(
      self,
      x: _typing.PyTree,
  ) -> tuple[jax.Array, _typing.PyTree]:
    """Evaluates `(f(X), grad_f1(X) + P diag(grad f2_vec(sigma(X))) Q^T)`."""
    mat = typing.cast(jax.Array, x)
    v1, g1_raw = self.f1_obj.value_and_grad(mat)
    g1 = typing.cast(jax.Array, g1_raw)
    if isinstance(self.f2_vec_obj, terms.ZeroSmooth):
      return v1, g1
    p_mat, s_vec, qt_mat = jnp.linalg.svd(mat, full_matrices=False)
    v2, gs_raw = self.f2_vec_obj.value_and_grad(s_vec)
    gs = typing.cast(jax.Array, gs_raw)
    return v1 + v2, g1 + reconstruct(p_mat, gs, qt_mat)

  def proximal_value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates `prox_vec(sigma(X))`."""
    mat = typing.cast(jax.Array, x)
    if isinstance(self.prox_vec, prox_base.Zero):
      return jnp.asarray(0.0, dtype=mat.dtype)
    s_vec = jnp.linalg.svdvals(mat)
    return self.prox_vec.value(s_vec)

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates `f1(X) + f2_vec(sigma(X)) + prox_vec(sigma(X))`."""
    mat = typing.cast(jax.Array, x)
    v1 = self.f1_obj.value(mat)
    need_f2 = not isinstance(self.f2_vec_obj, terms.ZeroSmooth)
    need_h = not isinstance(self.prox_vec, prox_base.Zero)
    if not need_f2 and not need_h:
      return v1
    s_vec = jnp.linalg.svdvals(mat)
    v2 = (
        self.f2_vec_obj.value(s_vec)
        if need_f2
        else jnp.asarray(0.0, dtype=mat.dtype)
    )
    vh = (
        self.prox_vec.value(s_vec)
        if need_h
        else jnp.asarray(0.0, dtype=mat.dtype)
    )
    return v1 + v2 + vh

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Evaluates spectral proximal mapping `P diag(prox_{tau h^V}(s)) Q^T`."""
    mat = typing.cast(jax.Array, x)
    if isinstance(self.prox_vec, prox_base.Zero):
      return mat
    p_mat, s_vec, qt_mat = jnp.linalg.svd(mat, full_matrices=False)
    s_prox = typing.cast(jax.Array, self.prox_vec.prox(s_vec, step_size))
    return reconstruct(p_mat, s_prox, qt_mat)

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Projects `X` onto `Omega` (if supplied) and `cl(dom(h))`."""
    mat = typing.cast(jax.Array, x)
    if self.domain_proj is not None:
      mat = self.domain_proj(mat)
    if isinstance(self.prox_vec, prox_base.Zero):
      return mat
    p_mat, s_vec, qt_mat = jnp.linalg.svd(mat, full_matrices=False)
    s_dom = typing.cast(jax.Array, self.prox_vec.domain_project(s_vec))
    return reconstruct(p_mat, s_dom, qt_mat)

  def omega_project(self, mat: jax.Array) -> jax.Array:
    """Applies `domain_proj` (`Pi_Omega`) if present, else `domain_project`."""
    if self.domain_proj is not None:
      return self.domain_proj(mat)
    return typing.cast(jax.Array, self.domain_project(mat))

  def f1_redistributed_vag(
      self,
      mat: jax.Array,
      xi: jax.Array,
  ) -> tuple[jax.Array, jax.Array]:
    """Evaluates `(f1(X) - 0.5 * xi * ||X||_F^2, grad_f1(X) - xi * X)`."""
    v1, g1_raw = self.f1_obj.value_and_grad(mat)
    g1 = typing.cast(jax.Array, g1_raw)
    return v1 - 0.5 * xi * jnp.sum(mat * mat), g1 - xi * mat

  def f1_redistributed_val(self, mat: jax.Array, xi: jax.Array) -> jax.Array:
    """Evaluates `f1(X) - 0.5 * xi * ||X||_F^2`."""
    return self.f1_obj.value(mat) - 0.5 * xi * jnp.sum(mat * mat)

  def f2_redistributed_val(self, s_vec: jax.Array, xi: jax.Array) -> jax.Array:
    """Evaluates `f2_vec(s) + 0.5 * xi * ||s||_2^2`."""
    return self.f2_vec_obj.value(s_vec) + 0.5 * xi * jnp.sum(s_vec * s_vec)

  def f2_redistributed_vag(
      self,
      s_vec: jax.Array,
      xi: jax.Array,
  ) -> tuple[jax.Array, jax.Array]:
    """Evaluates `(f2_tilde(s), grad f2_tilde(s))`."""
    v2, g2_raw = self.f2_vec_obj.value_and_grad(s_vec)
    g2 = typing.cast(jax.Array, g2_raw)
    return v2 + 0.5 * xi * jnp.sum(s_vec * s_vec), g2 + xi * s_vec

  def vector_subproblem_oracle(
      self,
      lam: jax.Array,
      x_hat: jax.Array,
      xi: jax.Array,
  ) -> VectorSpectralSubOracle:
    """Builds the 1D singular-value subproblem oracle for `ACG`."""
    return VectorSpectralSubOracle(
        f2_vec=self.f2_vec_obj,
        prox_vec=self.prox_vec,
        lam=lam,
        x_hat=x_hat,
        xi=xi,
    )


jax.tree_util.register_dataclass(
    SpectralOracle,
    data_fields=["f1", "f2_vec", "prox_vec"],
    meta_fields=["domain_proj"],
)


def delta_mu(
    oracle: SpectralOracle,
    *,
    lam: jax.Array,
    xi: jax.Array,
    mu: float,
    y0_mat: jax.Array,
    f1_y0: jax.Array,
    g1_y0: jax.Array,
    y_mat: jax.Array,
    y_vec: jax.Array,
    s_mat: jax.Array,
    s_vec: jax.Array,
    v_mat: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Computes `Delta_mu(S; Y, Y0, V)` in cancellation-free proximal units."""
  diff_s_y0 = s_mat - y0_mat
  diff_y_y0 = y_mat - y0_mat
  diff_y_s = y_mat - s_mat

  y0_norm_sq = jnp.sum(y0_mat * y0_mat)
  f1_orig_y0 = f1_y0 + 0.5 * xi * y0_norm_sq
  g1_orig_y0 = g1_y0 + xi * y0_mat

  f2h_orig_s = oracle.f2_vec_obj.value(s_vec) + oracle.prox_vec.value(s_vec)
  f2h_orig_y = oracle.f2_vec_obj.value(y_vec) + oracle.prox_vec.value(y_vec)

  dist_s_y0_sq = jnp.sum(diff_s_y0 * diff_s_y0)
  dist_y_y0_sq = jnp.sum(diff_y_y0 * diff_y_y0)
  dist_s_y_sq = jnp.sum(diff_y_s * diff_y_s)

  eff_prox = 0.5 * (1.0 + lam * xi)
  delta = (
      lam * (jnp.sum(g1_orig_y0 * diff_y_s) + (f2h_orig_y - f2h_orig_s))
      + eff_prox * (dist_y_y0_sq - dist_s_y0_sq)
      - jnp.sum(v_mat * diff_y_s)
      + 0.5 * mu * dist_s_y_sq
  )

  prox_at_s = (
      lam * (f1_orig_y0 + jnp.sum(g1_orig_y0 * diff_s_y0) + f2h_orig_s)
      + eff_prox * dist_s_y0_sq
      - jnp.sum(v_mat * s_mat)
      - 0.5 * mu * dist_s_y_sq
  )
  prox_at_y = prox_at_s + delta
  return delta, prox_at_s, prox_at_y


def refine_icg(
    oracle: SpectralOracle,
    *,
    p_mat: jax.Array,
    qt_mat: jax.Array,
    z0_mat: jax.Array,
    g1_z0: jax.Array,
    x0_vec: jax.Array,
    z_vec: jax.Array,
    v_vec: jax.Array,
    lam: jax.Array,
    xi: jax.Array,
    m2_est: jax.Array,
    jit: bool = True,
) -> SpectralRefinementResult:
  """Exact spectral refinement (`refine_ICG.m` with duplicate-`Q_hat` fix).

  Produces `Z_hat = P diag(z_hat) Q^T` and exact composite subgradient
  `V_hat in grad_f1(Z_hat) + grad_f2(Z_hat) + dh(Z_hat)` using only 1D vector
  proximal steps and a single matrix `f1` value-and-gradient evaluation.
  """
  dtype = z_vec.dtype
  f2_z, g2_z = oracle.f2_redistributed_vag(z_vec, xi)
  num_slack = 10.0 * tree.eps(dtype) * (1.0 + jnp.abs(f2_z))

  known_m2 = oracle.f2_vec_obj.curvature_upper
  if known_m2 is not None:
    m2_used = jnp.maximum(jnp.asarray(known_m2, dtype=dtype) + xi, 0.0)
    nu = lam / (lam * m2_used + 1.0)
    w_r = lam * g2_z - (v_vec + x0_vec - z_vec)
    z_ctr = z_vec - (nu / lam) * w_r
    z_hat = typing.cast(jax.Array, oracle.prox_vec.prox(z_ctr, nu))
    f2_zh, g2_zh = oracle.f2_redistributed_vag(z_hat, xi)
    vec_evals = jnp.asarray(2, dtype=jnp.int32)
  else:
    m2_init = jnp.maximum(m2_est, xi)

    def _vec_trial(m2_p: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
      nu_t = lam / (lam * m2_p + 1.0)
      w_t = lam * g2_z - (v_vec + x0_vec - z_vec)
      zc_t = z_vec - (nu_t / lam) * w_t
      zh_t = typing.cast(jax.Array, oracle.prox_vec.prox(zc_t, nu_t))
      f2_t = oracle.f2_redistributed_val(zh_t, xi)
      dz = zh_t - z_vec
      quad = f2_z + jnp.sum(g2_z * dz) + 0.5 * m2_p * jnp.sum(dz * dz)
      return zh_t, f2_t, f2_t <= quad + num_slack

    zh0, f20, ok0 = _vec_trial(m2_init)
    init_c = _VecRefineCarry(
        m2_plus=m2_init,
        z_hat=zh0,
        f2_z_hat=f20,
        accepted=ok0,
        trials=jnp.asarray(1, dtype=jnp.int32),
    )

    def _cond(c: _VecRefineCarry) -> jax.Array:
      return jnp.logical_and(jnp.logical_not(c.accepted), c.trials < 35)

    def _body(c: _VecRefineCarry) -> _VecRefineCarry:
      next_m2 = jnp.maximum(2.0 * c.m2_plus, 1e-6)
      zht, f2t, okt = _vec_trial(next_m2)
      return _VecRefineCarry(
          m2_plus=next_m2,
          z_hat=zht,
          f2_z_hat=f2t,
          accepted=okt,
          trials=c.trials + 1,
      )

    fin_c = _loop.while_loop(_cond, _body, init_c, jit=jit)
    m2_used = fin_c.m2_plus
    nu = lam / (lam * m2_used + 1.0)
    w_r = lam * g2_z - (v_vec + x0_vec - z_vec)
    z_ctr = z_vec - (nu / lam) * w_r
    z_hat = fin_c.z_hat
    f2_zh, g2_zh = oracle.f2_redistributed_vag(z_hat, xi)
    vec_evals = fin_c.trials + 2

  z_mat = reconstruct(p_mat, z_vec, qt_mat)
  v_mat = reconstruct(p_mat, v_vec, qt_mat)
  z_hat_mat = reconstruct(p_mat, z_hat, qt_mat)

  f1_orig_zh, g1_orig_zh_raw = oracle.f1_obj.value_and_grad(z_hat_mat)
  g1_orig_zh = typing.cast(jax.Array, g1_orig_zh_raw)
  g2_orig_zh = g2_zh - xi * z_hat
  subgrad_h_vec = (z_ctr - z_hat) / nu

  q_hat = (v_mat + z0_mat - z_mat) / lam + (z_mat - z_hat_mat) / nu
  v_hat = g1_orig_zh + reconstruct(p_mat, g2_orig_zh + subgrad_h_vec, qt_mat)

  f2_orig_zh = f2_zh - 0.5 * xi * jnp.sum(z_hat * z_hat)
  obj_zh = f1_orig_zh + f2_orig_zh + oracle.prox_vec.value(z_hat)
  return SpectralRefinementResult(
      z_hat_vec=z_hat,
      z_hat_mat=z_hat_mat,
      q_hat_mat=q_hat,
      v_hat_mat=v_hat,
      m2_used=m2_used,
      obj_val=obj_zh,
      f1_evals=jnp.asarray(1, dtype=jnp.int32),
      f1_grad_evals=jnp.asarray(1, dtype=jnp.int32),
      vec_evals=vec_evals,
  )


def __getattr__(name: str) -> typing.Any:
  if name in ("SpectralProx", "SpectralProblem", "as_spectral_oracle"):
    from proxax.oracle import spectral_problem  # noqa: PLC0415

    return getattr(spectral_problem, name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
