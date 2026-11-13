"""Adaptive Accelerated Composite Gradient (ACG) inner subproblem solver."""

from __future__ import annotations

import dataclasses
import enum

import jax
import jax.numpy as jnp

from proxax import _loop
from proxax import _typing
from proxax import tree
from proxax.oracle import composite
from proxax.solvers import base


class ACGStatus(enum.IntEnum):
  """Termination status for an inner `ACG` proximal subproblem run."""

  RUNNING = -1
  SIGMA_SATISFIED = 0
  NONCONVEX_DETECTED = 1
  MAX_ITER_REACHED = 2
  LINE_SEARCH_FAILED = 3


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ACGState(base.OracleCountsMixin):
  """Functional state of the estimating-sequence `ACG` method."""

  x: _typing.PyTree
  y: _typing.PyTree
  x0: _typing.PyTree
  psi_x0: jax.Array
  psi_y: jax.Array
  u: _typing.PyTree
  eta: jax.Array
  a_sum: jax.Array
  lipschitz: jax.Array
  iteration: jax.Array
  status: jax.Array
  counts: base.OracleCounts


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _ACGLineSearchCarry:
  """Internal backtracking carry for a single `ACG` step."""

  lipschitz: jax.Array
  a_step: jax.Array
  a_next: jax.Array
  x_tilde: _typing.PyTree
  y_next: _typing.PyTree
  f_y_next: jax.Array
  accepted: jax.Array
  backtracks: jax.Array


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ACGConfig:
  """Hyperparameters for the inner `ACG` subproblem solver."""

  sigma: float | jax.Array = 0.3
  rhs_offset_sq: float | jax.Array = 0.0
  max_iters: int = 1000
  backtrack_factor: float = 2.0
  ls_decrease: float = 0.9
  max_backtracks: int = 40
  lipschitz_min: float = 1e-12


def init_acg(
    oracle: composite.CompositeOracleProtocol,
    x_init: _typing.PyTree,
    lipschitz_init: _typing.Scalar,
) -> ACGState:
  """Initializes `ACGState` at `x0 = y0 = x_init`."""
  dtype = tree.infer_dtype(x_init)
  psi_x0 = oracle.value(x_init)
  L0 = jnp.maximum(
      jnp.asarray(lipschitz_init, dtype=dtype),
      jnp.asarray(1e-12, dtype=dtype),
  )
  return ACGState(
      x=x_init,
      y=x_init,
      x0=x_init,
      psi_x0=psi_x0,
      psi_y=psi_x0,
      u=tree.zeros_like(x_init),
      eta=jnp.asarray(0.0, dtype=dtype),
      a_sum=jnp.asarray(0.0, dtype=dtype),
      lipschitz=L0,
      iteration=jnp.asarray(0, dtype=jnp.int32),
      status=jnp.asarray(int(ACGStatus.RUNNING), dtype=jnp.int32),
      counts=base.OracleCounts.create(smooth=1),
  )


def step_acg(
    oracle: composite.CompositeOracleProtocol,
    state: ACGState,
    *,
    prox_center: _typing.PyTree,
    lam: jax.Array,
    config: ACGConfig,
    check_convexity: jax.Array | bool,
    jit: bool = True,
) -> ACGState:
  """Performs one adaptive `ACG` step with certificate and convexity checks."""
  dtype = state.lipschitz.dtype
  mu = jnp.asarray(oracle.strong_convexity, dtype=dtype)
  c_mu = 1.0 + mu * state.a_sum
  b_mu = 1.0 + 2.0 * mu * state.a_sum

  def _trial_from_lip(
      L_val: jax.Array,
  ) -> tuple[
      jax.Array,
      jax.Array,
      _typing.PyTree,
      _typing.PyTree,
      jax.Array,
      jax.Array,
  ]:
    disc = b_mu * b_mu + 4.0 * L_val * state.a_sum * c_mu
    a_step = (b_mu + jnp.sqrt(disc)) / (2.0 * L_val)
    a_next = state.a_sum + a_step
    d_mu = 1.0 + mu * a_next
    x_tilde = tree.convex_comb(
        state.a_sum * d_mu,
        state.y,
        a_step * c_mu,
        state.x,
    )
    f_tilde, grad_tilde = oracle.smooth_value_and_grad(x_tilde)
    inv_L = 1.0 / L_val
    y_next = oracle.prox(tree.add_scaled(x_tilde, -inv_L, grad_tilde), inv_L)
    f_y_next = oracle.smooth_value(y_next)
    diff_y = tree.sub(y_next, x_tilde)
    quad_upper = (
        f_tilde
        + tree.vdot(grad_tilde, diff_y)
        + 0.5 * L_val * tree.norm_sq(diff_y)
    )
    slack = 10.0 * tree.eps(dtype) * (1.0 + jnp.abs(f_tilde))
    ok = f_y_next <= quad_upper + slack
    return a_step, a_next, x_tilde, y_next, f_y_next, ok

  if oracle.curvature_upper is not None:
    L_used = jnp.maximum(
        jnp.asarray(oracle.curvature_upper, dtype=dtype),
        jnp.asarray(config.lipschitz_min, dtype=dtype),
    )
    a_step, a_next, x_tilde, y_next, f_y_next, ls_ok = _trial_from_lip(L_used)
    trials = jnp.asarray(1, dtype=jnp.int32)
    next_L = L_used
  else:
    L_start = jnp.maximum(
        state.lipschitz,
        jnp.asarray(config.lipschitz_min, dtype=dtype),
    )
    a0, an0, xt0, yn0, fyn0, ok0 = _trial_from_lip(L_start)
    init_ls = _ACGLineSearchCarry(
        lipschitz=L_start,
        a_step=a0,
        a_next=an0,
        x_tilde=xt0,
        y_next=yn0,
        f_y_next=fyn0,
        accepted=ok0,
        backtracks=jnp.asarray(0, dtype=jnp.int32),
    )

    def _ls_cond(c: _ACGLineSearchCarry) -> jax.Array:
      return jnp.logical_and(
          jnp.logical_not(c.accepted),
          c.backtracks < config.max_backtracks,
      )

    def _ls_body(c: _ACGLineSearchCarry) -> _ACGLineSearchCarry:
      L_try = c.lipschitz * config.backtrack_factor
      a_s, a_n, x_t, y_n, fy_n, ok = _trial_from_lip(L_try)
      return _ACGLineSearchCarry(
          lipschitz=L_try,
          a_step=a_s,
          a_next=a_n,
          x_tilde=x_t,
          y_next=y_n,
          f_y_next=fy_n,
          accepted=ok,
          backtracks=c.backtracks + 1,
      )

    fin_ls = _loop.while_loop(_ls_cond, _ls_body, init_ls, jit=jit)
    L_used = fin_ls.lipschitz
    a_step = fin_ls.a_step
    a_next = fin_ls.a_next
    x_tilde = fin_ls.x_tilde
    y_next = fin_ls.y_next
    f_y_next = fin_ls.f_y_next
    ls_ok = fin_ls.accepted
    trials = 1 + fin_ls.backtracks
    quad_curv = 0.5 * L_used * tree.norm_sq(tree.sub(y_next, x_tilde))
    noise_floor = 100.0 * tree.eps(dtype) * (1.0 + jnp.abs(f_y_next))
    relaxed_L = jnp.maximum(
        L_used * config.ls_decrease,
        jnp.asarray(config.lipschitz_min, dtype=dtype),
    )
    next_L = jnp.where(quad_curv > noise_floor, relaxed_L, L_used)

  psi_y_next = f_y_next + oracle.proximal_value(y_next)
  denom_mu = 1.0 + mu * a_next
  x_next = jax.tree.map(
      lambda xt, yt, x_til: (
          c_mu * xt + mu * a_step * yt - a_step * L_used * (x_til - yt)
      )
      / denom_mu,
      state.x,
      y_next,
      x_tilde,
  )

  # Exact (u_{t+1}, eta_{t+1}) in d_{eta_{t+1}} psi(y_{t+1})
  u_next = jax.tree.map(
      lambda x0_i, xn_i, yn_i: (
          (x0_i - xn_i) + mu * a_next * (yn_i - xn_i)
      )
      / a_next,
      state.x0,
      x_next,
      y_next,
  )
  dist_y_x0_sq = tree.norm_sq(tree.sub(y_next, state.x0))
  dist_y_xn_sq = tree.norm_sq(tree.sub(y_next, x_next))
  eta_raw = (dist_y_x0_sq - denom_mu * dist_y_xn_sq) / (2.0 * a_next)
  eta_next = jnp.maximum(jnp.asarray(0.0, dtype=dtype), eta_raw)

  # Check convexity invariant A_{t+1} (psi(x0) - psi(y_{t+1})) >= ...
  cvx_slack = (
      100.0
      * tree.eps(dtype)
      * (1.0 + jnp.abs(state.psi_x0))
      * (1.0 + a_next)
  )
  lhs_cvx = a_next * (state.psi_x0 - psi_y_next) + cvx_slack
  rhs_cvx = 0.5 * denom_mu * tree.norm_sq(tree.sub(state.x0, x_next))
  nonconvex_hit = jnp.logical_and(
      jnp.asarray(check_convexity),
      lhs_cvx < rhs_cvx,
  )

  # Check sigma-relative proximal error criterion:
  # ||u||^2 + (2 / lam) * eta <= sigma^2 * ||(prox_center - y) / lam + u||^2
  lhs_sigma = tree.norm_sq(u_next) + (2.0 / lam) * eta_next
  rhs_vec = tree.add(
      tree.scale(1.0 / lam, tree.sub(prox_center, y_next)),
      u_next,
  )
  eps_sq = tree.eps(dtype) ** 2
  rhs_norm_sq = tree.norm_sq(rhs_vec) + jnp.asarray(
      config.rhs_offset_sq, dtype=dtype
  )
  rhs_sigma = (config.sigma**2) * rhs_norm_sq + eps_sq
  sigma_ok = lhs_sigma <= rhs_sigma

  next_iter = state.iteration + 1
  status = jnp.where(
      jnp.logical_not(ls_ok),
      jnp.asarray(int(ACGStatus.LINE_SEARCH_FAILED), dtype=jnp.int32),
      jnp.where(
          nonconvex_hit,
          jnp.asarray(int(ACGStatus.NONCONVEX_DETECTED), dtype=jnp.int32),
          jnp.where(
              sigma_ok,
              jnp.asarray(int(ACGStatus.SIGMA_SATISFIED), dtype=jnp.int32),
              jnp.where(
                  next_iter >= config.max_iters,
                  jnp.asarray(int(ACGStatus.MAX_ITER_REACHED), dtype=jnp.int32),
                  jnp.asarray(int(ACGStatus.RUNNING), dtype=jnp.int32),
              ),
          ),
      ),
  )

  return ACGState(
      x=x_next,
      y=y_next,
      x0=state.x0,
      psi_x0=state.psi_x0,
      psi_y=psi_y_next,
      u=u_next,
      eta=eta_next,
      a_sum=a_next,
      lipschitz=next_L,
      iteration=next_iter,
      status=status,
      counts=state.counts.add(
          smooth=2 * trials,
          grad=trials,
          prox=trials,
      ),
  )


def run_acg(
    oracle: composite.CompositeOracleProtocol,
    x_init: _typing.PyTree,
    *,
    prox_center: _typing.PyTree,
    lam: jax.Array,
    lipschitz_init: _typing.Scalar,
    config: ACGConfig,
    check_convexity: jax.Array | bool = False,
    jit: bool = True,
) -> ACGState:
  """Runs `ACG` until the sigma-criterion, nonconvexity, or iteration limit."""
  init_state = init_acg(oracle, x_init, lipschitz_init)
  running_code = jnp.asarray(int(ACGStatus.RUNNING), dtype=jnp.int32)

  def _cond(st: ACGState) -> jax.Array:
    return st.status == running_code

  def _body(st: ACGState) -> ACGState:
    return step_acg(
        oracle,
        st,
        prox_center=prox_center,
        lam=lam,
        config=config,
        check_convexity=check_convexity,
        jit=jit,
    )

  return _loop.while_loop(_cond, _body, init_state, jit=jit)
