"""Certified proximal ADMM for block-constrained composite optimization."""

from __future__ import annotations

import dataclasses
import time
import typing

import jax
import jax.numpy as jnp

from proxax import _typing
from proxax import tree
from proxax.oracle import block as block_mod
from proxax.oracle import terms
from proxax.solvers import base


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ADMM:
  """Certified proximal ADMM for ``BlockProblem``.

  Solves::

      min_{x_1, ..., x_K}  sum_i (f_i(x_i) + h_i(x_i))
      s.t.                sum_i A_i(x_i) = rhs

  Uses exact proximal updates for blocks with ``smooth=0, linear_op=identity``
  and adaptive Lipschitz-backtracking linearized proximal updates otherwise,
  returning exact primal-dual KKT stationarity and feasibility certificates.

  Attributes:
    opt_tol: Target KKT stationarity tolerance ``||v|| <= opt_tol``.
    feas_tol: Target coupling feasibility tolerance ``||r|| <= feas_tol``.
    rho: Augmented Lagrangian penalty parameter ``rho > 0``.
    max_iters: Maximum outer ADMM iterations.
    lipschitz_init: Initial local smoothness estimate for general blocks.
    backtrack_factor: Multiplicative Lipschitz growth factor (> 1).
    max_backtracks: Maximum backtracking trials per block per iteration.
    norm_kind: Norm type (``'l2'`` or ``'linf'``) for residual checks.
  """

  opt_tol: float = 1e-5
  feas_tol: float = 1e-5
  rho: float = 1.0
  max_iters: int = 2000
  lipschitz_init: float = 1.0
  backtrack_factor: float = 2.0
  max_backtracks: int = 30
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
    """Solves ``problem`` starting from block tuple ``x0``."""
    x_init = tuple(
        blk.proximal.domain_project(x_i)
        for blk, x_i in zip(problem.blocks, x0, strict=True)
    )
    bound_ops = problem.bind_operators(x_init)
    dtype = tree.infer_dtype(x_init[0])
    zero_val = jnp.asarray(0.0, dtype=dtype)
    inf_val = jnp.asarray(jnp.inf, dtype=dtype)

    ax_init = problem.coupling_residual(x_init, bound_ops)
    y_init = tree.zeros_like(ax_init)
    lips_init = tuple(
        jnp.asarray(self.lipschitz_init, dtype=dtype) for _ in problem.blocks
    )
    v_init = tuple(tree.zeros_like(x_i) for x_i in x_init)
    counts_init = base.OracleCounts.zeros()

    def _step(
        state: tuple[
            tuple[_typing.PyTree, ...],
            _typing.PyTree,
            tuple[jax.Array, ...],
            _typing.PyTree,
            tuple[_typing.PyTree, ...],
            jax.Array,
            jax.Array,
            jax.Array,
            base.OracleCounts,
        ],
    ) -> tuple[
        tuple[_typing.PyTree, ...],
        _typing.PyTree,
        tuple[jax.Array, ...],
        _typing.PyTree,
        tuple[_typing.PyTree, ...],
        jax.Array,
        jax.Array,
        jax.Array,
        base.OracleCounts,
    ]:
      x_cur, y_cur, lips_cur, ax_cur, _, _, _, it_cur, counts_cur = state
      rho_val = jnp.asarray(self.rho, dtype=dtype)
      x_next_list: list[_typing.PyTree] = []
      lips_next_list: list[jax.Array] = []
      taus_list: list[jax.Array] = []
      grads_used_list: list[_typing.PyTree] = []
      ax_running = ax_cur
      smooth_evals = counts_cur.smooth_evals
      grad_evals = counts_cur.grad_evals
      prox_evals = counts_cur.prox_evals

      for blk, op, x_i, lip_i in zip(
          problem.blocks, bound_ops, x_cur, lips_cur, strict=True
      ):
        if blk.is_zero_smooth and blk.is_identity_op:
          tau_i = 1.0 / rho_val
          g_i = tree.add(y_cur, tree.scale(rho_val, ax_running))
          target_i = tree.sub(x_i, tree.scale(tau_i, g_i))
          x_i_next = blk.proximal.prox(target_i, tau_i)
          ax_running = tree.add(ax_running, tree.sub(x_i_next, x_i))
          lip_i_next = lip_i
          prox_evals = prox_evals + 1
        else:
          smooth_i = typing.cast(terms.SmoothObjective, blk.smooth)
          ax_i_cur = op.matvec(x_i)
          r_shift = tree.add(
              tree.sub(ax_running, ax_i_cur),
              tree.scale(1.0 / rho_val, y_cur),
          )
          f_val_i, f_grad_i = smooth_i.value_and_grad(x_i)
          grad_evals = grad_evals + 1
          aug_vec = tree.add(y_cur, tree.scale(rho_val, ax_running))
          g_i = tree.add(f_grad_i, op.rmatvec(aug_vec))
          psi_cur = f_val_i + 0.5 * rho_val * tree.norm_sq(
              tree.add(ax_i_cur, r_shift)
          )

          lip_try = jnp.maximum(lip_i / 1.2, jnp.asarray(1e-6, dtype=dtype))
          bt_init = (
              lip_try,
              x_i,
              ax_i_cur,
              jnp.asarray(False),
              jnp.asarray(0, dtype=jnp.int32),
          )

          _BTState = tuple[
              jax.Array,
              _typing.PyTree,
              _typing.PyTree,
              jax.Array,
              jax.Array,
          ]

          def _bt_cond(bt_st: _BTState) -> jax.Array:
            _, _, _, ok, bt_it = bt_st
            return (~ok) & (bt_it < self.max_backtracks)

          def _bt_body(
              bt_st: _BTState,
              _x_i: _typing.PyTree = x_i,
              _g_i: _typing.PyTree = g_i,
              _psi_cur: jax.Array = psi_cur,
              _r_shift: _typing.PyTree = r_shift,
              _smooth_i: terms.SmoothObjective = smooth_i,
              _blk: block_mod.BlockTerm = blk,
              _op: typing.Any = op,
          ) -> _BTState:
            l_cur, _, _, _, bt_it = bt_st
            tau = 1.0 / l_cur
            cand = _blk.proximal.prox(
                tree.sub(_x_i, tree.scale(tau, _g_i)), tau
            )
            ax_cand = _op.matvec(cand)
            psi_cand = _smooth_i.value(cand) + 0.5 * rho_val * tree.norm_sq(
                tree.add(ax_cand, _r_shift)
            )
            dx = tree.sub(cand, _x_i)
            rhs_val = (
                _psi_cur
                + tree.vdot(_g_i, dx)
                + 0.5 * l_cur * tree.norm_sq(dx)
            )
            accepted = psi_cand <= rhs_val + 1e-10 * (
                1.0 + jnp.abs(_psi_cur)
            )
            l_next = jnp.where(
                accepted, l_cur, l_cur * self.backtrack_factor
            )
            return l_next, cand, ax_cand, accepted, bt_it + 1

          lip_i_next, x_i_next, ax_i_next, _, bt_iters = jax.lax.while_loop(
              _bt_cond, _bt_body, bt_init
          )
          tau_i = 1.0 / lip_i_next
          ax_running = tree.add(ax_running, tree.sub(ax_i_next, ax_i_cur))
          smooth_evals = smooth_evals + bt_iters
          prox_evals = prox_evals + bt_iters

        x_next_list.append(x_i_next)
        lips_next_list.append(lip_i_next)
        taus_list.append(tau_i)
        grads_used_list.append(g_i)

      x_next = tuple(x_next_list)
      y_next = tree.add(y_cur, tree.scale(rho_val, ax_running))

      # Exact KKT stationarity residuals:
      # v_i = grad f_i(x_i^+) + A_i^*(y^+) + (x_i - x_i^+) / tau_i - g_i
      v_list: list[_typing.PyTree] = []
      for blk, op, x_i, x_i_next, tau_i, g_i in zip(
          problem.blocks,
          bound_ops,
          x_cur,
          x_next,
          taus_list,
          grads_used_list,
          strict=True,
      ):
        smooth_i = typing.cast(terms.SmoothObjective, blk.smooth)
        if blk.is_zero_smooth:
          f_grad_next = tree.zeros_like(x_i_next)
        else:
          _, f_grad_next = smooth_i.value_and_grad(x_i_next)
          grad_evals = grad_evals + 1
        a_adj_y = op.rmatvec(y_next)
        prox_subgrad = tree.sub(
            tree.scale(1.0 / tau_i, tree.sub(x_i, x_i_next)), g_i
        )
        v_i = tree.add(tree.add(f_grad_next, a_adj_y), prox_subgrad)
        v_list.append(v_i)

      v_next = tuple(v_list)
      stat_next = tree.norm(v_next, kind=self.norm_kind)
      feas_next = tree.norm(ax_running, kind=self.norm_kind)
      counts_next = base.OracleCounts(
          smooth_evals=smooth_evals,
          grad_evals=grad_evals,
          prox_evals=prox_evals,
      )
      return (
          x_next,
          y_next,
          tuple(lips_next_list),
          ax_running,
          v_next,
          stat_next,
          feas_next,
          it_cur + 1,
          counts_next,
      )

    init_state = (
        x_init,
        y_init,
        lips_init,
        ax_init,
        v_init,
        inf_val,
        inf_val,
        jnp.asarray(0, dtype=jnp.int32),
        counts_init,
    )

    def _not_done(
        state: tuple[
            tuple[_typing.PyTree, ...],
            _typing.PyTree,
            tuple[jax.Array, ...],
            _typing.PyTree,
            tuple[_typing.PyTree, ...],
            jax.Array,
            jax.Array,
            jax.Array,
            base.OracleCounts,
        ],
    ) -> jax.Array:
      _, _, _, _, _, stat_cur, feas_cur, it_cur, _ = state
      converged = (stat_cur <= self.opt_tol) & (feas_cur <= self.feas_tol)
      return (~converged) & (it_cur < self.max_iters)

    t0 = time.perf_counter()
    if jit and callback is None and time_limit is None:
      final_state = jax.lax.while_loop(_not_done, _step, init_state)
      timed_out = False
    else:
      final_state = init_state
      timed_out = False
      while bool(_not_done(final_state)):
        if time_limit is not None and (time.perf_counter() - t0) >= time_limit:
          timed_out = True
          break
        final_state = _step(final_state)
        if callback is not None:
          callback(int(final_state[7]), final_state[5], final_state[6])

    x_fin, y_fin, _, _, v_fin, stat_fin, feas_fin, it_fin, counts_fin = (
        final_state
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
        dual=y_fin,
        subgrad=v_fin,
        status=status,
        stats=base.SolveStats(
            outer_iters=it_fin,
            inner_iters=zero_val.astype(jnp.int32),
            smooth_evals=counts_fin.smooth_evals,
            grad_evals=counts_fin.grad_evals,
            prox_evals=counts_fin.prox_evals,
            opt_residual=stat_fin,
            feas_residual=feas_fin,
            obj_val=obj_val,
        ),
    )
