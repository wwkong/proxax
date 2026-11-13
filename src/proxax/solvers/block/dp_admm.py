"""Dampened Proximal ADMM (DP.ADMM, Kong & Monteiro 2024, SIOPT)."""

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
class _DPBlockSmooth:
  """Smooth part of block `t`'s cyclic `DP.ADMM` proximal subproblem."""

  base_smooth: terms.SmoothObjective
  linear_op: typing.Any
  shift_vec: _typing.PyTree
  damp_p: _typing.PyTree
  x_prev_t: _typing.PyTree
  lam: jax.Array
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
        + tree.vdot(self.damp_p, au)
        + 0.5 * self.c_val * tree.norm_sq(res)
    )
    diff = tree.sub(u, self.x_prev_t)
    return self.lam * aug + 0.5 * tree.norm_sq(diff)

  def value_and_grad(
      self,
      u: _typing.PyTree,
  ) -> tuple[jax.Array, _typing.PyTree]:
    f_val, f_grad = self.base_smooth.value_and_grad(u)
    au = self.linear_op.matvec(u)
    res = tree.add(au, self.shift_vec)
    aug_val = (
        f_val
        + tree.vdot(self.damp_p, au)
        + 0.5 * self.c_val * tree.norm_sq(res)
    )
    aug_grad = tree.add(
        f_grad,
        self.linear_op.rmatvec(
            tree.add(self.damp_p, tree.scale(self.c_val, res))
        ),
    )
    diff = tree.sub(u, self.x_prev_t)
    val = self.lam * aug_val + 0.5 * tree.norm_sq(diff)
    grad = tree.add(tree.scale(self.lam, aug_grad), diff)
    return val, grad


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _ScaledProx:
  """Scaled proximal operator `lam * h_t(u)`."""

  base_prox: prox_base.ProxOperator
  lam: jax.Array

  @property
  def strong_convexity(self) -> float:
    return 0.0

  def value(self, x: _typing.PyTree) -> jax.Array:
    return self.lam * self.base_prox.value(x)

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    return self.base_prox.prox(x, step_size * self.lam)

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    return self.base_prox.domain_project(x)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DPADMM:
  """Dampened Proximal ADMM (`DP.ADMM`, Kong & Monteiro 2024, arXiv:2110.12502).

  Cyclic Gauss-Seidel proximal ADMM with dampened multiplier updates
  ``p^k = (1 - theta) p^{k-1} + chi * c * (A x^k - b)`` and adaptive
  penalty doubling ``c <- 2 c`` across stages, achieving ``O(eps^{-3})``
  iteration complexity without rank assumptions.

  Attributes:
    opt_tol: Target KKT stationarity tolerance ``||v|| <= opt_tol``.
    feas_tol: Target feasibility tolerance ``||Ax - b|| <= feas_tol``.
    theta: Multiplier dampening factor in ``(0, 1)``.
    chi: Multiplier step factor in ``(0, 1]``; defaults to the theoretical
      threshold ``theta^2 / (2 * B * (2 - theta) * (1 - theta))``.
    lam_init: Initial proximal stepsize ``lambda > 0`` (automatically halved
      if `PF.ACG` detects non-strong-convexity).
    c_init: Initial augmented Lagrangian penalty parameter ``c_0 > 0``.
    max_stages: Maximum penalty-doubling stages.
    max_stage_iters: Maximum static `DP.ADMM` iterations per stage.
    max_inner_iters: Maximum `PF.ACG` iterations per block subproblem.
    norm_kind: Norm type (``'l2'`` or ``'linf'``) for residual checks.
  """

  opt_tol: float = 1e-5
  feas_tol: float = 1e-5
  theta: float = 0.5
  chi: float | None = None
  lam_init: float = 1.0
  c_init: float = 1.0
  max_stages: int = 30
  max_stage_iters: int = 200
  max_inner_iters: int = 200
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
    """Solves ``problem`` via dynamic `DP.ADMM` starting from ``x0``."""
    num_blocks = problem.num_blocks
    x_init = tuple(
        blk.proximal.domain_project(x_i)
        for blk, x_i in zip(problem.blocks, x0, strict=True)
    )
    bound_ops = problem.bind_operators(x_init)
    dtype = tree.infer_dtype(x_init[0])
    zero_val = jnp.asarray(0.0, dtype=dtype)
    zero_int = jnp.asarray(0, dtype=jnp.int32)

    theta_val = jnp.asarray(self.theta, dtype=dtype)
    chi_bound = (self.theta**2) / (
        2.0 * max(num_blocks, 1) * (2.0 - self.theta) * (1.0 - self.theta)
    )
    chi_val = jnp.asarray(
        self.chi if self.chi is not None else chi_bound, dtype=dtype
    )

    ax_init = problem.coupling_residual(x_init, bound_ops)
    p_init = tree.zeros_like(ax_init)
    v_init = tuple(tree.zeros_like(x_i) for x_i in x_init)
    pf_cfg = pf_acg.PFACGConfig(
        mu=0.5,
        sigma=0.25,
        max_iters=self.max_inner_iters,
    )
    bad_code = jnp.asarray(
        int(pf_acg.PFACGStatus.BAD_TERMINATION), dtype=jnp.int32
    )

    _StageState = tuple[
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
        base.OracleCounts,
    ]

    def _run_stage(st: _StageState) -> _StageState:
      x_s0, p_s0, _, lam_s0, c_cur, _, _, s_it, o_it, in_it, cnt_cur = st

      _StaticState = tuple[
          tuple[_typing.PyTree, ...],
          _typing.PyTree,
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

      def _st_cond(sst: _StaticState) -> jax.Array:
        _, _, _, _, _, stat_cur, feas_cur, k_it, _, _, done, _, _ = sst
        conv = (stat_cur <= self.opt_tol) & (feas_cur <= self.feas_tol)
        return (~conv) & (~done) & (k_it < self.max_stage_iters)

      def _st_body(sst: _StaticState) -> _StaticState:
        (
            x_prev,
            p_prev,
            _,
            _,
            lam_cur,
            _,
            _,
            k_cur,
            o_cur,
            in_cur,
            _,
            _,
            cnt_prev,
        ) = sst
        damp_p = tree.scale(1.0 - theta_val, p_prev)
        ax_running = problem.coupling_residual(x_prev, bound_ops)

        x_next_list: list[_typing.PyTree] = []
        dax_list: list[_typing.PyTree] = []
        w_list: list[_typing.PyTree] = []
        lam_next = lam_cur
        smooth_add = zero_int
        grad_add = zero_int
        prox_add = zero_int
        inner_add = zero_int

        for blk, op, x_t in zip(
            problem.blocks, bound_ops, x_prev, strict=True
        ):
          ax_t_prev = op.matvec(x_t)
          shift_vec = tree.sub(ax_running, ax_t_prev)
          smooth_t = typing.cast(terms.SmoothObjective, blk.smooth)

          def _solve_block(
              lam_try: jax.Array,
              _blk: block_mod.BlockTerm = blk,
              _op: typing.Any = op,
              _x_t: _typing.PyTree = x_t,
              _shift: _typing.PyTree = shift_vec,
              _smooth_t: terms.SmoothObjective = smooth_t,
          ) -> tuple[
              jax.Array,
              _typing.PyTree,
              _typing.PyTree,
              jax.Array,
              base.OracleCounts,
              jax.Array,
          ]:
            sub_smooth = _DPBlockSmooth(
                base_smooth=_smooth_t,
                linear_op=_op,
                shift_vec=_shift,
                damp_p=damp_p,
                x_prev_t=_x_t,
                lam=lam_try,
                c_val=c_cur,
            )
            sub_prox = _ScaledProx(base_prox=_blk.proximal, lam=lam_try)
            sub_oracle = composite.CompositeOracle(
                smooth=sub_smooth, proximal=sub_prox
            )
            pf_st = pf_acg.run_pf_acg(
                sub_oracle,
                _x_t,
                lipschitz_init=1.0 + lam_try * c_cur,
                config=pf_cfg,
                jit=jit,
            )
            is_bad = pf_st.status == bad_code
            return (
                is_bad,
                pf_st.y,
                pf_st.r,
                pf_st.iteration,
                pf_st.counts,
                lam_try,
            )

          def _lam_cond(
              lt_st: tuple[
                  jax.Array,
                  _typing.PyTree,
                  _typing.PyTree,
                  jax.Array,
                  base.OracleCounts,
                  jax.Array,
                  jax.Array,
              ],
          ) -> jax.Array:
            is_bad, _, _, _, _, _, tries = lt_st
            return is_bad & (tries < 15)

          def _lam_body(
              lt_st: tuple[
                  jax.Array,
                  _typing.PyTree,
                  _typing.PyTree,
                  jax.Array,
                  base.OracleCounts,
                  jax.Array,
                  jax.Array,
              ],
              _solve: typing.Any = _solve_block,
          ) -> tuple[
              jax.Array,
              _typing.PyTree,
              _typing.PyTree,
              jax.Array,
              base.OracleCounts,
              jax.Array,
              jax.Array,
          ]:
            _, _, _, in_t, c_t, lam_v, tries = lt_st
            next_lam = jnp.maximum(0.5 * lam_v, jnp.asarray(1e-8, dtype=dtype))
            bad, y_v, r_v, in_new, c_new, _ = _solve(next_lam)
            tot_c = base.OracleCounts(
                smooth_evals=c_t.smooth_evals + c_new.smooth_evals,
                grad_evals=c_t.grad_evals + c_new.grad_evals,
                prox_evals=c_t.prox_evals + c_new.prox_evals,
            )
            return bad, y_v, r_v, in_t + in_new, tot_c, next_lam, tries + 1

          bad0, x_t_next, w_t, in0, cnt0, _ = _solve_block(lam_next)
          _, x_t_next, w_t, in_fin_t, cnt_fin_t, lam_next, _ = (
              jax.lax.while_loop(
                  _lam_cond,
                  _lam_body,
                  (bad0, x_t_next, w_t, in0, cnt0, lam_next, zero_int),
              )
          )
          dax_t = tree.sub(op.matvec(x_t_next), ax_t_prev)
          ax_running = tree.add(ax_running, dax_t)
          x_next_list.append(x_t_next)
          dax_list.append(dax_t)
          w_list.append(w_t)
          smooth_add = smooth_add + cnt_fin_t.smooth_evals
          grad_add = grad_add + cnt_fin_t.grad_evals
          prox_add = prox_add + cnt_fin_t.prox_evals
          inner_add = inner_add + in_fin_t

        x_next = tuple(x_next_list)
        q_next = tree.add(damp_p, tree.scale(c_cur, ax_running))

        # Exact KKT stationarity residuals:
        # v_t = (w_t - dx_t) / lam + c A_t^* sum_{s>t} A_s(dx_s)
        suffix_dax = tree.zeros_like(ax_running)
        v_rev: list[_typing.PyTree] = []
        inv_lam = 1.0 / lam_next
        for op, x_t, x_t_next, w_t, dax_t in zip(
            reversed(bound_ops),
            reversed(x_prev),
            reversed(x_next),
            reversed(w_list),
            reversed(dax_list),
            strict=True,
        ):
          dz_t = tree.sub(x_t_next, x_t)
          v_t = tree.add(
              tree.sub(tree.scale(inv_lam, w_t), tree.scale(inv_lam, dz_t)),
              tree.scale(c_cur, op.rmatvec(suffix_dax)),
          )
          v_rev.append(v_t)
          suffix_dax = tree.add(suffix_dax, dax_t)
        v_next = tuple(reversed(v_rev))

        stat_next = tree.norm(v_next, kind=self.norm_kind)
        feas_next = tree.norm(ax_running, kind=self.norm_kind)
        p_next = tree.add(damp_p, tree.scale(chi_val * c_cur, ax_running))

        # STEP 2b: Double penalty c only once stationarity is small relative to
        # feasibility (or when the stage iteration cap is reached).
        k_next = k_cur + 1
        restart_check = (k_next >= 10) & (stat_next <= self.opt_tol) & (
            feas_next > self.feas_tol
        )
        cnt_next = base.OracleCounts(
            smooth_evals=cnt_prev.smooth_evals + smooth_add,
            grad_evals=cnt_prev.grad_evals + grad_add,
            prox_evals=cnt_prev.prox_evals + prox_add,
        )
        return (
            x_next,
            p_next,
            q_next,
            v_next,
            lam_next,
            stat_next,
            feas_next,
            k_next,
            o_cur + 1,
            in_cur + inner_add,
            restart_check,
            zero_val,
            cnt_next,
        )

      init_st: _StaticState = (
          x_s0,
          p_s0,
          p_s0,
          v_init,
          lam_s0,
          jnp.asarray(jnp.inf, dtype=dtype),
          jnp.asarray(jnp.inf, dtype=dtype),
          zero_int,
          o_it,
          in_it,
          jnp.asarray(False),
          zero_val,
          cnt_cur,
      )
      fin_st = jax.lax.while_loop(_st_cond, _st_body, init_st)
      (
          x_fin_s,
          p_fin_s,
          q_fin_s,
          v_fin_s,
          lam_fin_s,
          stat_s,
          feas_s,
          _,
          o_fin,
          in_fin,
          _,
          _,
          cnt_fin,
      ) = fin_st
      del q_fin_s
      return (
          x_fin_s,
          p_fin_s,
          v_fin_s,
          lam_fin_s,
          2.0 * c_cur,
          stat_s,
          feas_s,
          s_it + 1,
          o_fin,
          in_fin,
          cnt_fin,
      )

    init_stage: _StageState = (
        x_init,
        p_init,
        v_init,
        jnp.asarray(self.lam_init, dtype=dtype),
        jnp.asarray(self.c_init, dtype=dtype),
        jnp.asarray(jnp.inf, dtype=dtype),
        jnp.asarray(jnp.inf, dtype=dtype),
        zero_int,
        zero_int,
        zero_int,
        base.OracleCounts.zeros(),
    )

    def _stage_cond(st: _StageState) -> jax.Array:
      _, _, _, _, _, stat_s, feas_s, s_it, _, _, _ = st
      done = (stat_s <= self.opt_tol) & (feas_s <= self.feas_tol)
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
          callback(int(final_stage[7]), final_stage[5], final_stage[6])

    x_fin, p_fin, v_fin, _, _, stat_fin, feas_fin, _, o_fin, in_fin, cnt_fin = (
        final_stage
    )
    converged = (stat_fin <= self.opt_tol) & (feas_fin <= self.feas_tol)
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
