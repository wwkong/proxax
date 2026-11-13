"""Parallel Adaptive Proximal ADMM (P-Adapt-ADMM, Maia–Monteiro–Silva 2026)."""

from __future__ import annotations

import dataclasses
import time
import typing

import jax
import jax.numpy as jnp

from proxax import _typing
from proxax import tree
from proxax.oracle import block as block_mod
from proxax.oracle import composite
from proxax.oracle import terms
from proxax.prox import base as prox_base
from proxax.solvers import base
from proxax.solvers.inner import pf_acg


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _BlockSubSmooth:
  """Smooth part of block `t`'s `P-ABIPP` subproblem."""

  base_smooth: terms.SmoothObjective
  linear_op: typing.Any
  shift_vec: _typing.PyTree
  q_vec: _typing.PyTree
  y_t: _typing.PyTree
  lam_t: jax.Array
  c_val: jax.Array

  @property
  def curvature_upper(self) -> None:
    return None

  @property
  def curvature_lower(self) -> None:
    return None

  def value(self, u: _typing.PyTree) -> jax.Array:
    au = self.linear_op.matvec(u)
    res = tree.add(au, self.shift_vec)
    aug = (
        self.base_smooth.value(u)
        + tree.vdot(self.q_vec, au)
        + 0.5 * self.c_val * tree.norm_sq(res)
    )
    diff = tree.sub(u, self.y_t)
    return self.lam_t * aug + 0.5 * tree.norm_sq(diff)

  def value_and_grad(
      self,
      u: _typing.PyTree,
  ) -> tuple[jax.Array, _typing.PyTree]:
    f_val, f_grad = self.base_smooth.value_and_grad(u)
    au = self.linear_op.matvec(u)
    res = tree.add(au, self.shift_vec)
    aug_val = (
        f_val
        + tree.vdot(self.q_vec, au)
        + 0.5 * self.c_val * tree.norm_sq(res)
    )
    aug_grad = tree.add(
        f_grad,
        self.linear_op.rmatvec(
            tree.add(self.q_vec, tree.scale(self.c_val, res))
        ),
    )
    diff = tree.sub(u, self.y_t)
    val = self.lam_t * aug_val + 0.5 * tree.norm_sq(diff)
    grad = tree.add(tree.scale(self.lam_t, aug_grad), diff)
    return val, grad


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _ScaledProx:
  """Scaled proximal operator `lam_t * h_t(u)`."""

  base_prox: prox_base.ProxOperator
  lam_t: jax.Array

  @property
  def strong_convexity(self) -> float:
    return 0.0

  def value(self, x: _typing.PyTree) -> jax.Array:
    return self.lam_t * self.base_prox.value(x)

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    return self.base_prox.prox(x, step_size * self.lam_t)

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    return self.base_prox.domain_project(x)


def _aug_lagrangian(
    problem: block_mod.BlockProblem,
    bound_ops: tuple[typing.Any, ...],
    u_blocks: tuple[_typing.PyTree, ...],
    q_vec: _typing.PyTree,
    c_val: jax.Array,
) -> jax.Array:
  """Evaluates the augmented Lagrangian ``L_c(u; q)``."""
  res = problem.coupling_residual(u_blocks, bound_ops)
  return (
      problem.objective_value(u_blocks)
      + tree.vdot(q_vec, res)
      + 0.5 * c_val * tree.norm_sq(res)
  )


def _solve_block_with_halving(
    problem: block_mod.BlockProblem,
    bound_ops: tuple[typing.Any, ...],
    t_idx: int,
    blk: block_mod.BlockTerm,
    op: typing.Any,
    y_blocks: tuple[_typing.PyTree, ...],
    y_t: _typing.PyTree,
    shift_vec: _typing.PyTree,
    q_vec: _typing.PyTree,
    lam_t: jax.Array,
    c_val: jax.Array,
    lc_y: jax.Array,
    tau_bar: jax.Array,
    pf_cfg: pf_acg.PFACGConfig,
    good_code: jax.Array,
    opt_code: jax.Array,
    max_lam_halvings: int,
    jit: bool,
    dtype: typing.Any,
) -> tuple[
    jax.Array,
    _typing.PyTree,
    _typing.PyTree,
    base.OracleCounts,
    jax.Array,
]:
  """Solves one block subproblem with backtracking on proximal step size."""
  smooth_t = typing.cast(terms.SmoothObjective, blk.smooth)
  num_blocks = len(y_blocks)
  zero_int = jnp.asarray(0, dtype=jnp.int32)

  def _halve_cond(
      st: tuple[
          jax.Array,
          _typing.PyTree,
          _typing.PyTree,
          jax.Array,
          jax.Array,
          base.OracleCounts,
          jax.Array,
      ],
  ) -> jax.Array:
    _, _, _, ok, it_h, _, _ = st
    return (~ok) & (it_h < max_lam_halvings)

  def _halve_body(
      st: tuple[
          jax.Array,
          _typing.PyTree,
          _typing.PyTree,
          jax.Array,
          jax.Array,
          base.OracleCounts,
          jax.Array,
      ],
  ) -> tuple[
      jax.Array,
      _typing.PyTree,
      _typing.PyTree,
      jax.Array,
      jax.Array,
      base.OracleCounts,
      jax.Array,
  ]:
    lam_cur, _, _, _, it_h, cnt_cur, in_cur = st
    sub_smooth = _BlockSubSmooth(
        base_smooth=smooth_t,
        linear_op=op,
        shift_vec=shift_vec,
        q_vec=q_vec,
        y_t=y_t,
        lam_t=lam_cur,
        c_val=c_val,
    )
    sub_prox = _ScaledProx(base_prox=blk.proximal, lam_t=lam_cur)
    sub_oracle = composite.CompositeOracle(
        smooth=sub_smooth,
        proximal=sub_prox,
    )
    pf_st = pf_acg.run_pf_acg(
        sub_oracle,
        y_t,
        lipschitz_init=1.0 + lam_cur * c_val,
        config=pf_cfg,
        jit=jit,
    )
    z_cand = pf_st.y
    w_cand = pf_st.r
    pf_ok = (pf_st.status == good_code) | (pf_st.status == opt_code)

    # Check block sufficient-decrease condition (eq. 14).
    dz = tree.sub(z_cand, y_t)
    y_trial = tree.add(y_t, tree.scale(tau_bar, dz))
    trial_blocks = tuple(
        y_trial if j == t_idx else y_blocks[j] for j in range(num_blocks)
    )
    lc_trial = _aug_lagrangian(problem, bound_ops, trial_blocks, q_vec, c_val)
    d_az = op.matvec(dz)
    req_dec = ((tau_bar**2) / (2.0 * lam_cur)) * tree.norm_sq(dz) + (
        0.25 * c_val * tau_bar
    ) * tree.norm_sq(d_az)
    dec_ok = (lc_y - lc_trial) >= req_dec - 1e-10 * (1.0 + jnp.abs(lc_y))
    ok = pf_ok & dec_ok
    lam_min = jnp.asarray(1e-8, dtype=dtype)
    lam_next = jnp.where(ok, lam_cur, jnp.maximum(0.5 * lam_cur, lam_min))
    cnt_next = base.OracleCounts(
        smooth_evals=cnt_cur.smooth_evals + pf_st.counts.smooth_evals,
        grad_evals=cnt_cur.grad_evals + pf_st.counts.grad_evals,
        prox_evals=cnt_cur.prox_evals + pf_st.counts.prox_evals,
    )
    return (
        lam_next,
        z_cand,
        w_cand,
        ok,
        it_h + 1,
        cnt_next,
        in_cur + pf_st.iteration,
    )

  init_h = (
      lam_t,
      y_t,
      tree.zeros_like(y_t),
      jnp.asarray(False),
      zero_int,
      base.OracleCounts.zeros(),
      zero_int,
  )
  lam_fin, z_t, w_t, _, _, cnt_t, in_t = jax.lax.while_loop(
      _halve_cond, _halve_body, init_h
  )
  return lam_fin, z_t, w_t, cnt_t, in_t


def _bisect_tau(
    problem: block_mod.BlockProblem,
    bound_ops: tuple[typing.Any, ...],
    y_blocks: tuple[_typing.PyTree, ...],
    z_blocks: tuple[_typing.PyTree, ...],
    lam_next: tuple[jax.Array, ...],
    q_vec: _typing.PyTree,
    c_val: jax.Array,
    lc_y: jax.Array,
    tau_bar: jax.Array,
    dtype: typing.Any,
) -> jax.Array:
  """Runs Appendix B's 8-step bisection search for relaxation ``tau``."""
  dz_blocks = tuple(
      tree.sub(z_t, y_t) for y_t, z_t in zip(y_blocks, z_blocks, strict=True)
  )
  dz_lam_sum = sum(
      (
          tree.norm_sq(dz_t) / lam_t
          for dz_t, lam_t in zip(dz_blocks, lam_next, strict=True)
      ),
      jnp.asarray(0.0, dtype=dtype),
  )
  daz_sum = sum(
      (
          tree.norm_sq(op.matvec(dz_t))
          for op, dz_t in zip(bound_ops, dz_blocks, strict=True)
      ),
      jnp.asarray(0.0, dtype=dtype),
  )

  def _bisect_body(
      _: int,
      bounds: tuple[jax.Array, jax.Array],
  ) -> tuple[jax.Array, jax.Array]:
    tau_lo, tau_hi = bounds
    tau_mid = 0.5 * (tau_lo + tau_hi)
    y_mid = tuple(
        tree.add(y_t, tree.scale(tau_mid, dz_t))
        for y_t, dz_t in zip(y_blocks, dz_blocks, strict=True)
    )
    lc_mid = _aug_lagrangian(problem, bound_ops, y_mid, q_vec, c_val)
    req = (
        0.5 * (tau_mid**2) * dz_lam_sum
        + (0.125 * c_val * tau_mid) * daz_sum
    )
    admissible = (lc_y - lc_mid) >= req - 1e-10 * (1.0 + jnp.abs(lc_y))
    return (
        jnp.where(admissible, tau_mid, tau_lo),
        jnp.where(admissible, tau_hi, tau_mid),
    )

  tau_used, _ = jax.lax.fori_loop(
      0,
      8,
      _bisect_body,
      (tau_bar, jnp.asarray(1.0, dtype=dtype)),
  )
  return typing.cast(jax.Array, tau_used)


def _p_abipp_step(
    problem: block_mod.BlockProblem,
    bound_ops: tuple[typing.Any, ...],
    y_blocks: tuple[_typing.PyTree, ...],
    q_vec: _typing.PyTree,
    lam_blocks: tuple[jax.Array, ...],
    c_val: jax.Array,
    *,
    tau_bar: jax.Array,
    pf_cfg: pf_acg.PFACGConfig,
    good_code: jax.Array,
    opt_code: jax.Array,
    max_lam_halvings: int,
    adaptive_tau: bool,
    jit: bool,
    dtype: typing.Any,
) -> tuple[
    tuple[_typing.PyTree, ...],
    tuple[jax.Array, ...],
    tuple[_typing.PyTree, ...],
    tuple[_typing.PyTree, ...],
    jax.Array,
    jax.Array,
    base.OracleCounts,
]:
  """Executes one parallel `P-ABIPP` block step and returns KKT residual."""
  zero_int = jnp.asarray(0, dtype=jnp.int32)
  zero_val = jnp.asarray(0.0, dtype=dtype)
  ay_list = [
      op.matvec(y_t) for op, y_t in zip(bound_ops, y_blocks, strict=True)
  ]
  ay_res = problem.coupling_residual(y_blocks, bound_ops)
  lc_y = _aug_lagrangian(problem, bound_ops, y_blocks, q_vec, c_val)

  z_next_list: list[_typing.PyTree] = []
  w_next_list: list[_typing.PyTree] = []
  lam_next_list: list[jax.Array] = []
  total_smooth = zero_int
  total_grad = zero_int
  total_prox = zero_int
  total_inner = zero_int

  for t_idx, (blk, op, y_t, ay_t, lam_t) in enumerate(
      zip(
          problem.blocks,
          bound_ops,
          y_blocks,
          ay_list,
          lam_blocks,
          strict=True,
      )
  ):
    shift_vec = tree.sub(ay_res, ay_t)
    lam_fin, z_t, w_t, cnt_t, in_t = _solve_block_with_halving(
        problem,
        bound_ops,
        t_idx,
        blk,
        op,
        y_blocks,
        y_t,
        shift_vec,
        q_vec,
        lam_t,
        c_val,
        lc_y,
        tau_bar,
        pf_cfg,
        good_code,
        opt_code,
        max_lam_halvings,
        jit,
        dtype,
    )
    z_next_list.append(z_t)
    w_next_list.append(w_t)
    lam_next_list.append(lam_fin)
    total_smooth = total_smooth + cnt_t.smooth_evals
    total_grad = total_grad + cnt_t.grad_evals
    total_prox = total_prox + cnt_t.prox_evals
    total_inner = total_inner + in_t

  z_blocks = tuple(z_next_list)
  lam_next = tuple(lam_next_list)
  if adaptive_tau:
    tau_used = _bisect_tau(
        problem,
        bound_ops,
        y_blocks,
        z_blocks,
        lam_next,
        q_vec,
        c_val,
        lc_y,
        tau_bar,
        dtype,
    )
  else:
    tau_used = tau_bar

  y_next = tuple(
      tree.add(y_t, tree.scale(tau_used, tree.sub(z_t, y_t)))
      for y_t, z_t in zip(y_blocks, z_blocks, strict=True)
  )

  # Step 2: Compute exact coupled KKT residual r^+.
  az_diff_total = tree.sub(
      problem.coupling_residual(z_blocks, bound_ops),
      ay_res,
  )
  r_list: list[_typing.PyTree] = []
  for op, y_t, z_t, w_t, lam_t in zip(
      bound_ops, y_blocks, z_blocks, w_next_list, lam_next, strict=True
  ):
    dz_t = tree.sub(z_t, y_t)
    daz_other = tree.sub(az_diff_total, op.matvec(dz_t))
    inv_lam = 1.0 / lam_t
    r_t = tree.add(
        tree.sub(tree.scale(inv_lam, w_t), tree.scale(inv_lam, dz_t)),
        tree.scale(c_val, op.rmatvec(daz_other)),
    )
    r_list.append(r_t)

  return (
      y_next,
      lam_next,
      z_blocks,
      tuple(r_list),
      total_inner,
      zero_val,
      base.OracleCounts(
          smooth_evals=total_smooth,
          grad_evals=total_grad,
          prox_evals=total_prox,
      ),
  )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class PAdaptADMM:
  """Parallel Adaptive Proximal ADMM (`P-Adapt-ADMM`, arXiv:2611.00570).

  Solves separable multi-block linearly constrained composite problems::

      min_{x_1, ..., x_B}  sum_t (f_t(x_t) + h_t(x_t))
      s.t.                sum_t A_t(x_t) = rhs

  via Jacobi-type parallel block updates (`P-ABIPP` backed by `PF.ACG`),
  adaptive block proximal stepsizes `lambda_t`, a Lagrange Multiplier Test
  (LMT), and geometric penalty updates `c_s = 2 * c_{s-1}`.

  Attributes:
    opt_tol: Target KKT stationarity tolerance ``||r||_2 <= opt_tol``.
    feas_tol: Target feasibility tolerance ``||Ax - b||_2 <= feas_tol``.
    alpha: LMT balancing parameter ``alpha > 0``.
    lmt_bound: Stationarity threshold ``C >= opt_tol`` for LMT activation.
    gamma_init: Initial proximal stepsize ``gamma_0 > 0`` per block.
    sigma: Relative stationarity tolerance for `PF.ACG` (default ``1/32``).
    max_stages: Maximum outer penalty-doubling stages.
    max_stage_iters: Maximum `P-ABIPP` calls per stage.
    max_inner_iters: Maximum `PF.ACG` iterations per block subproblem.
    max_lam_halvings: Maximum proximal stepsize halvings per `P-ABIPP` call.
    adaptive_tau: Whether to run Appendix B's 8-step bisection search for the
      relaxation parameter ``tau in [1/(4(B-1)), 1]`` (practical variant).
    norm_kind: Norm type (``'l2'`` or ``'linf'``) for stopping checks.
  """

  opt_tol: float = 1e-5
  feas_tol: float = 1e-5
  alpha: float = 1.0
  lmt_bound: float | None = None
  gamma_init: float = 1.0
  sigma: float = 1.0 / 32.0
  max_stages: int = 30
  max_stage_iters: int = 200
  max_inner_iters: int = 200
  max_lam_halvings: int = 25
  adaptive_tau: bool = False
  norm_kind: str = "l2"

  def solve(
      self,
      problem: block_mod.BlockProblem,
      x0: tuple[_typing.PyTree, ...],
      *,
      jit: bool = True,
      callback: typing.Callable[[int, jax.Array, jax.Array], None] | None = (
          None
      ),
      time_limit: float | None = None,
  ) -> base.Solution:
    """Solves ``problem`` via `P-Adapt-ADMM` starting from ``x0``."""
    num_blocks = problem.num_blocks
    if num_blocks < 2:
      raise ValueError("PAdaptADMM requires at least 2 blocks.")

    x_init = tuple(
        blk.proximal.domain_project(x_i)
        for blk, x_i in zip(problem.blocks, x0, strict=True)
    )
    bound_ops = problem.bind_operators(x_init)
    dtype = tree.infer_dtype(x_init[0])
    zero_val = jnp.asarray(0.0, dtype=dtype)
    zero_int = jnp.asarray(0, dtype=jnp.int32)
    tau_bar = jnp.asarray(1.0 / (4.0 * (num_blocks - 1)), dtype=dtype)
    rho_sq = jnp.asarray(self.opt_tol * self.opt_tol, dtype=dtype)
    lmt_val = (
        self.lmt_bound if self.lmt_bound is not None else max(1.0, self.opt_tol)
    )
    lmt_c_sq = jnp.asarray(lmt_val * lmt_val, dtype=dtype)
    alpha_val = jnp.asarray(self.alpha, dtype=dtype)

    ax_init = problem.coupling_residual(x_init, bound_ops)
    p_init = tree.zeros_like(ax_init)
    c_init = 1.0 / (1.0 + tree.norm(ax_init, kind="l2"))
    gamma_0 = tuple(
        jnp.asarray(self.gamma_init, dtype=dtype) for _ in problem.blocks
    )
    v_init = tuple(tree.zeros_like(x_i) for x_i in x_init)
    pf_cfg = pf_acg.PFACGConfig(
        mu=0.5,
        sigma=self.sigma,
        max_iters=self.max_inner_iters,
    )
    good_code = jnp.asarray(
        int(pf_acg.PFACGStatus.GOOD_TERMINATION), dtype=jnp.int32
    )
    opt_code = jnp.asarray(
        int(pf_acg.PFACGStatus.OPT_TOL_SATISFIED), dtype=jnp.int32
    )

    # Outer stage loop state:
    # (x_s, p_s, v_s, gamma_s, c_s, feas_s, stat_s, stage_it, outer_it,
    #  inner_it, counts)
    _StageState = tuple[
        tuple[_typing.PyTree, ...],
        _typing.PyTree,
        tuple[_typing.PyTree, ...],
        tuple[jax.Array, ...],
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        base.OracleCounts,
    ]

    def _run_stage(st: _StageState) -> _StageState:
      x_s0, p_s0, _, gamma_s0, c_cur, _, _, s_it, o_it, in_it, cnt_cur = st

      # Inner loop state within stage:
      # (y_i, q_i, lam_i, z_out, q_out, r_out, t_sum, max_dq_sq, sum_dq_sq,
      #  sum_res_sq, i_it, done, o_it, in_it, cnt)
      _InnerState = tuple[
          tuple[_typing.PyTree, ...],
          _typing.PyTree,
          tuple[jax.Array, ...],
          tuple[_typing.PyTree, ...],
          _typing.PyTree,
          tuple[_typing.PyTree, ...],
          jax.Array,
          jax.Array,
          jax.Array,
          jax.Array,
          jax.Array,
          jax.Array,
          jax.Array,
          jax.Array,
          base.OracleCounts,
      ]

      def _in_cond(ist: _InnerState) -> jax.Array:
        _, _, _, _, _, _, _, _, _, _, i_it, done, _, _, _ = ist
        return (~done) & (i_it < self.max_stage_iters)

      def _in_body(ist: _InnerState) -> _InnerState:
        (
            y_prev,
            q_prev,
            lam_prev,
            _,
            _,
            _,
            t_prev,
            max_dq_prev,
            sum_dq_prev,
            sum_res_prev,
            i_prev,
            _,
            o_prev,
            in_prev,
            cnt_prev,
        ) = ist
        y_i, lam_i, z_i, r_i, inner_delta, _, cnt_delta = _p_abipp_step(
            problem,
            bound_ops,
            y_prev,
            q_prev,
            lam_prev,
            c_cur,
            tau_bar=tau_bar,
            pf_cfg=pf_cfg,
            good_code=good_code,
            opt_code=opt_code,
            max_lam_halvings=self.max_lam_halvings,
            adaptive_tau=self.adaptive_tau,
            jit=jit,
            dtype=dtype,
        )
        az_res = problem.coupling_residual(z_i, bound_ops)
        q_cand = tree.add(q_prev, tree.scale(c_cur, az_res))
        r_sq = tree.norm_sq(r_i)
        stat_ok = r_sq <= rho_sq

        t_i = t_prev + (
            _aug_lagrangian(problem, bound_ops, y_prev, q_prev, c_cur)
            - _aug_lagrangian(problem, bound_ops, y_i, q_prev, c_cur)
        )
        dq_sq = tree.norm_sq(tree.sub(q_cand, q_prev))
        max_dq_cand = jnp.maximum(max_dq_prev, dq_sq)
        sum_dq_cand = sum_dq_prev + dq_sq
        r_cap = sum_dq_cand / jnp.maximum(
            max_dq_cand, jnp.asarray(1e-30, dtype=dtype)
        )
        i_next = i_prev + 1
        sum_res_next = sum_res_prev + r_sq
        avg_res = sum_res_next / i_next.astype(dtype)
        lmt_ok = (r_sq <= lmt_c_sq) & (avg_res >= t_i / (alpha_val * r_cap))

        q_next = jax.tree.map(
            lambda qc, qp: jnp.where(stat_ok | lmt_ok, qc, qp),
            q_cand,
            q_prev,
        )
        max_dq_next = jnp.where(lmt_ok, max_dq_cand, max_dq_prev)
        sum_dq_next = jnp.where(lmt_ok, sum_dq_cand, sum_dq_prev)
        cnt_next = base.OracleCounts(
            smooth_evals=cnt_prev.smooth_evals + cnt_delta.smooth_evals,
            grad_evals=cnt_prev.grad_evals + cnt_delta.grad_evals,
            prox_evals=cnt_prev.prox_evals + cnt_delta.prox_evals,
        )
        return (
            y_i,
            q_next,
            lam_i,
            z_i,
            q_cand,
            r_i,
            t_i,
            max_dq_next,
            sum_dq_next,
            sum_res_next,
            i_next,
            stat_ok,
            o_prev + 1,
            in_prev + inner_delta,
            cnt_next,
        )

      init_in: _InnerState = (
          x_s0,
          p_s0,
          gamma_s0,
          x_s0,
          p_s0,
          v_init,
          zero_val,
          zero_val,
          zero_val,
          zero_val,
          zero_int,
          jnp.asarray(False),
          o_it,
          in_it,
          cnt_cur,
      )
      fin_in = jax.lax.while_loop(_in_cond, _in_body, init_in)
      _, _, lam_s, z_s, q_s, r_s, _, _, _, _, _, _, o_fin, in_fin, cnt_fin = (
          fin_in
      )
      feas_s = tree.norm(
          problem.coupling_residual(z_s, bound_ops), kind=self.norm_kind
      )
      stat_s = tree.norm(r_s, kind=self.norm_kind)
      return (
          z_s,
          q_s,
          r_s,
          lam_s,
          2.0 * c_cur,
          feas_s,
          stat_s,
          s_it + 1,
          o_fin,
          in_fin,
          cnt_fin,
      )

    init_stage: _StageState = (
        x_init,
        p_init,
        v_init,
        gamma_0,
        c_init,
        jnp.asarray(jnp.inf, dtype=dtype),
        jnp.asarray(jnp.inf, dtype=dtype),
        zero_int,
        zero_int,
        zero_int,
        base.OracleCounts.zeros(),
    )

    def _stage_cond(st: _StageState) -> jax.Array:
      _, _, _, _, _, feas_s, stat_s, s_it, _, _, _ = st
      done = (feas_s <= self.feas_tol) & (stat_s <= self.opt_tol)
      return (~done) & (s_it < self.max_stages)

    t0 = time.perf_counter()
    if jit and callback is None and time_limit is None:
      final_stage = jax.lax.while_loop(_stage_cond, _run_stage, init_stage)
      timed_out = False
    else:
      final_stage = init_stage
      timed_out = False
      while bool(_stage_cond(final_stage)):
        if time_limit is not None and (time.perf_counter() - t0) >= time_limit:
          timed_out = True
          break
        final_stage = _run_stage(final_stage)
        if callback is not None:
          callback(int(final_stage[7]), final_stage[6], final_stage[5])

    x_fin, p_fin, v_fin, _, _, feas_fin, stat_fin, _, o_fin, in_fin, cnt_fin = (
        final_stage
    )
    converged = (feas_fin <= self.feas_tol) & (stat_fin <= self.opt_tol)
    if timed_out:
      status = jnp.asarray(
          base.SolveStatus.TIME_LIMIT_REACHED, dtype=jnp.int32
      )
    else:
      status = jnp.where(
          converged,
          jnp.asarray(base.SolveStatus.CONVERGED, dtype=jnp.int32),
          jnp.asarray(base.SolveStatus.MAX_ITER_REACHED, dtype=jnp.int32),
      )

    obj_val = problem.objective_value(x_fin)
    del t0
    return base.Solution(
        x=x_fin,
        dual=p_fin,
        subgrad=v_fin,
        status=status,
        stats=base.SolveStats(
            outer_iters=o_fin,
            inner_iters=in_fin,
            smooth_evals=cnt_fin.smooth_evals,
            grad_evals=cnt_fin.grad_evals,
            prox_evals=cnt_fin.prox_evals,
            opt_residual=stat_fin,
            feas_residual=feas_fin,
            obj_val=obj_val,
        ),
    )
