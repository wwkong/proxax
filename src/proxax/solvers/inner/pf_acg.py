"""Parameter-Free Accelerated Composite Gradient (`PF.ACG`) inner solver."""

from __future__ import annotations

import dataclasses
import enum

import jax
import jax.numpy as jnp

from proxax import _loop
from proxax import _typing
from proxax import tree
from proxax.oracle import composite


class PFACGStatus(enum.IntEnum):
  """Termination status for a `PF.ACG` proximal subproblem run."""

  RUNNING = -1
  GOOD_TERMINATION = 0
  OPT_TOL_SATISFIED = 1
  BAD_TERMINATION = 2
  MAX_ITER_REACHED = 3
  LINE_SEARCH_FAILED = 4


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class PFACGState:
  """Functional state of `PF.ACG` (Algorithms 3.1--3.2 of Kong 2024)."""

  x: _typing.PyTree
  y: _typing.PyTree
  y0: _typing.PyTree
  psi_y0: jax.Array
  psi_y: jax.Array
  r: _typing.PyTree
  a_sum: jax.Array
  lipschitz: jax.Array
  iteration: jax.Array
  status: jax.Array
  smooth_evals: jax.Array
  grad_evals: jax.Array
  prox_evals: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _PFACGLineSearchCarry:
  """Internal backtracking carry for Algorithm 3.1 of `PF.ACG`."""

  lipschitz: jax.Array
  a_step: jax.Array
  a_next: jax.Array
  x_tilde: _typing.PyTree
  grad_tilde: _typing.PyTree
  y_next: _typing.PyTree
  f_y_next: jax.Array
  grad_y_next: _typing.PyTree
  accepted: jax.Array
  backtracks: jax.Array


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class PFACGConfig:
  """Hyperparameters for `PF.ACG` (Algorithms 3.1--3.2 of `apd.pdf`)."""

  mu: float = 0.5
  sigma: float = 0.5
  theta: float = 4.0
  beta: float = 2.0
  max_iters: int = 1000
  max_backtracks: int = 40
  lipschitz_min: float = 0.5


def init_pf_acg(
    oracle: composite.CompositeOracleProtocol,
    y_init: _typing.PyTree,
    lipschitz_init: _typing.Scalar,
    *,
    mu: float = 0.5,
) -> PFACGState:
  """Initializes `PFACGState` at `x0 = y0 = y_init` with `A_0 = 0`."""
  dtype = tree.infer_dtype(y_init)
  psi_y0 = oracle.value(y_init)
  L0 = jnp.maximum(
      jnp.asarray(lipschitz_init, dtype=dtype),
      jnp.asarray(mu, dtype=dtype),
  )
  return PFACGState(
      x=y_init,
      y=y_init,
      y0=y_init,
      psi_y0=psi_y0,
      psi_y=psi_y0,
      r=tree.zeros_like(y_init),
      a_sum=jnp.asarray(0.0, dtype=dtype),
      lipschitz=L0,
      iteration=jnp.asarray(0, dtype=jnp.int32),
      status=jnp.asarray(int(PFACGStatus.RUNNING), dtype=jnp.int32),
      smooth_evals=jnp.asarray(1, dtype=jnp.int32),
      grad_evals=jnp.asarray(0, dtype=jnp.int32),
      prox_evals=jnp.asarray(0, dtype=jnp.int32),
  )


def step_pf_acg(
    oracle: composite.CompositeOracleProtocol,
    state: PFACGState,
    *,
    config: PFACGConfig,
    two_m: jax.Array | None = None,
    opt_tol: jax.Array | None = None,
    norm_kind: str = "l2",
    jit: bool = True,
) -> PFACGState:
  """Performs one `PF.ACG` step with exact certificate `r_{j+1}` and checks.

  Implements Algorithm 3.1 (line search + accelerated composite gradient step)
  and Algorithm 3.2 Steps 2--3 (dynamic bad check Eq. (3.5) and good check
  Eq. (3.6)) from W. Kong, SIAM J. Optim. 34(3):3005--3032, 2024.

  Args:
    oracle: Rescaled subproblem `CompositeOracleProtocol` `(psi^s, psi^n)`.
    state: Current `PFACGState`.
    config: `PFACGConfig` hyperparameters.
    two_m: Optional scalar `2 * m_hat` enabling early outer stationarity check
      `||2 * m_hat * (r_{j+1} + y_0 - y_{j+1})|| <= opt_tol`.
    opt_tol: Optional target stationarity tolerance paired with `two_m`.
    norm_kind: Residual norm (`'l2'` or `'linf'`).
    jit: Whether to run line search via `jax.lax.while_loop` (`True`) or Python.

  Returns:
    Updated `PFACGState` with exact `r in grad_psi^s(y) + d psi^n(y)`.
  """
  dtype = state.lipschitz.dtype
  mu = jnp.asarray(config.mu, dtype=dtype)
  xi = 1.0 + mu * state.a_sum

  def _trial_from_lip(
      L_val: jax.Array,
  ) -> tuple[
      jax.Array,
      jax.Array,
      _typing.PyTree,
      _typing.PyTree,
      _typing.PyTree,
      jax.Array,
      _typing.PyTree,
      jax.Array,
  ]:
    disc = xi * xi + 4.0 * L_val * xi * state.a_sum
    a_step = (xi + jnp.sqrt(disc)) / (2.0 * L_val)
    a_next = state.a_sum + a_step
    x_tilde = tree.convex_comb(state.a_sum, state.y, a_step, state.x)
    f_tilde, grad_tilde = oracle.smooth_value_and_grad(x_tilde)
    inv_L_mu = 1.0 / (L_val + mu)
    y_next = oracle.prox(
        tree.add_scaled(x_tilde, -inv_L_mu, grad_tilde),
        inv_L_mu,
    )
    f_y_next, grad_y_next = oracle.smooth_value_and_grad(y_next)
    diff_y = tree.sub(y_next, x_tilde)
    quad_upper = (
        f_tilde
        + tree.vdot(grad_tilde, diff_y)
        + 0.5 * L_val * tree.norm_sq(diff_y)
    )
    slack = 10.0 * tree.eps(dtype) * (1.0 + jnp.abs(f_tilde))
    ok = f_y_next <= quad_upper + slack
    return (
        a_step,
        a_next,
        x_tilde,
        grad_tilde,
        y_next,
        f_y_next,
        grad_y_next,
        ok,
    )

  L_min = jnp.maximum(mu, jnp.asarray(config.lipschitz_min, dtype=dtype))
  L_start = jnp.maximum(state.lipschitz, L_min)
  a0, an0, xt0, gt0, yn0, fyn0, gyn0, ok0 = _trial_from_lip(L_start)
  init_ls = _PFACGLineSearchCarry(
      lipschitz=L_start,
      a_step=a0,
      a_next=an0,
      x_tilde=xt0,
      grad_tilde=gt0,
      y_next=yn0,
      f_y_next=fyn0,
      grad_y_next=gyn0,
      accepted=ok0,
      backtracks=jnp.asarray(0, dtype=jnp.int32),
  )

  def _ls_cond(c: _PFACGLineSearchCarry) -> jax.Array:
    return jnp.logical_and(
        jnp.logical_not(c.accepted),
        c.backtracks < config.max_backtracks,
    )

  def _ls_body(c: _PFACGLineSearchCarry) -> _PFACGLineSearchCarry:
    L_try = c.lipschitz * config.beta
    a_s, a_n, x_t, g_t, y_n, fy_n, gy_n, ok = _trial_from_lip(L_try)
    return _PFACGLineSearchCarry(
        lipschitz=L_try,
        a_step=a_s,
        a_next=a_n,
        x_tilde=x_t,
        grad_tilde=g_t,
        y_next=y_n,
        f_y_next=fy_n,
        grad_y_next=gy_n,
        accepted=ok,
        backtracks=c.backtracks + 1,
    )

  fin_ls = _loop.while_loop(_ls_cond, _ls_body, init_ls, jit=jit)
  L_used = fin_ls.lipschitz
  a_step = fin_ls.a_step
  a_next = fin_ls.a_next
  x_tilde = fin_ls.x_tilde
  grad_tilde = fin_ls.grad_tilde
  y_next = fin_ls.y_next
  f_y_next = fin_ls.f_y_next
  grad_y_next = fin_ls.grad_y_next
  ls_ok = fin_ls.accepted
  trials = 1 + fin_ls.backtracks

  psi_y_next = f_y_next + oracle.proximal_value(y_next)
  denom_mu = 1.0 + mu * a_next
  x_next = jax.tree.map(
      lambda xj, yn, xt: xj
      + (a_step / denom_mu) * (L_used * (yn - xt) + mu * (yn - xj)),
      state.x,
      y_next,
      x_tilde,
  )

  # Exact certificate r_{j+1} in grad_psi^s(y_{j+1}) + d psi^n(y_{j+1})
  r_next = tree.add(
      tree.sub(grad_y_next, grad_tilde),
      tree.scale(L_used + mu, tree.sub(x_tilde, y_next)),
  )

  diff_y0_y = tree.sub(state.y0, y_next)
  r_plus_y0_minus_y = tree.add(r_next, diff_y0_y)
  if two_m is not None and opt_tol is not None:
    v_cand = tree.scale(two_m, r_plus_y0_minus_y)
    opt_ok = tree.norm(v_cand, kind=norm_kind) <= opt_tol
  else:
    opt_ok = jnp.asarray(False, dtype=jnp.bool_)

  # Step 2 ("Bad" check, Eq. (3.5)): local mu-strong convexity invariants.
  eps_val = tree.eps(dtype)
  eps_sq = eps_val**2
  dist_y_xt_sq = tree.norm_sq(tree.sub(y_next, x_tilde))
  dist_y_y0_sq = tree.norm_sq(diff_y0_y)
  dist_slack = 100.0 * eps_val * dist_y_y0_sq + eps_sq
  bad_cond_1_ok = mu * a_next * dist_y_xt_sq <= dist_y_y0_sq + dist_slack

  rhs_subgrad = psi_y_next + tree.vdot(r_next, diff_y0_y)
  psi_scale = 1.0 + jnp.maximum(jnp.abs(state.psi_y0), jnp.abs(rhs_subgrad))
  cvx_slack = 100.0 * eps_val * psi_scale
  bad_cond_2_ok = state.psi_y0 + cvx_slack >= rhs_subgrad
  bad_hit = jnp.logical_not(jnp.logical_and(bad_cond_1_ok, bad_cond_2_ok))

  # Step 3 ("Good" check, Eq. (3.6)): proximal descent & error conditions.
  r_norm_sq = tree.norm_sq(r_next)
  good_cond_1 = r_norm_sq <= (config.sigma**2) * dist_y_y0_sq + eps_sq
  desc_slack = 100.0 * eps_val * (1.0 + jnp.abs(state.psi_y0))
  good_cond_2 = (
      tree.norm_sq(r_plus_y0_minus_y)
      <= config.theta
      * (state.psi_y0 - psi_y_next + 0.5 * dist_y_y0_sq + desc_slack)
      + eps_sq
  )
  good_ok = jnp.logical_and(good_cond_1, good_cond_2)

  next_iter = state.iteration + 1
  status = jnp.where(
      jnp.logical_not(ls_ok),
      jnp.asarray(int(PFACGStatus.LINE_SEARCH_FAILED), dtype=jnp.int32),
      jnp.where(
          opt_ok,
          jnp.asarray(int(PFACGStatus.OPT_TOL_SATISFIED), dtype=jnp.int32),
          jnp.where(
              bad_hit,
              jnp.asarray(int(PFACGStatus.BAD_TERMINATION), dtype=jnp.int32),
              jnp.where(
                  good_ok,
                  jnp.asarray(
                      int(PFACGStatus.GOOD_TERMINATION),
                      dtype=jnp.int32,
                  ),
                  jnp.where(
                      next_iter >= config.max_iters,
                      jnp.asarray(
                          int(PFACGStatus.MAX_ITER_REACHED),
                          dtype=jnp.int32,
                      ),
                      jnp.asarray(int(PFACGStatus.RUNNING), dtype=jnp.int32),
                  ),
              ),
          ),
      ),
  )

  return PFACGState(
      x=x_next,
      y=y_next,
      y0=state.y0,
      psi_y0=state.psi_y0,
      psi_y=psi_y_next,
      r=r_next,
      a_sum=a_next,
      lipschitz=L_used,
      iteration=next_iter,
      status=status,
      smooth_evals=state.smooth_evals + 2 * trials,
      grad_evals=state.grad_evals + 2 * trials,
      prox_evals=state.prox_evals + trials,
  )


def run_pf_acg(
    oracle: composite.CompositeOracleProtocol,
    y_init: _typing.PyTree,
    *,
    lipschitz_init: _typing.Scalar,
    config: PFACGConfig,
    two_m: jax.Array | None = None,
    opt_tol: jax.Array | None = None,
    norm_kind: str = "l2",
    jit: bool = True,
) -> PFACGState:
  """Runs `PF.ACG` until good/bad termination, stationarity, or iter cap."""
  init_state = init_pf_acg(oracle, y_init, lipschitz_init, mu=config.mu)
  running_code = jnp.asarray(int(PFACGStatus.RUNNING), dtype=jnp.int32)

  def _cond(st: PFACGState) -> jax.Array:
    return st.status == running_code

  def _body(st: PFACGState) -> PFACGState:
    return step_pf_acg(
        oracle,
        st,
        config=config,
        two_m=two_m,
        opt_tol=opt_tol,
        norm_kind=norm_kind,
        jit=jit,
    )

  return _loop.while_loop(_cond, _body, init_state, jit=jit)
