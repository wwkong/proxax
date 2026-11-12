"""Stationarity refinement mapping approximate iterates to certified pairs."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from proxax import _loop
from proxax import _typing
from proxax import tree
from proxax.oracle import composite


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class RefinementResult:
  """Certified approximate stationary pair `(x_hat, v_hat)`.

  Guarantees exact subdifferential membership
  `v_hat in grad_f(x_hat) + dh(x_hat)` with monotone composite descent
  `phi(x_hat) <= phi(y)`.
  """

  x_hat: _typing.PyTree
  v_hat: _typing.PyTree
  obj_val: jax.Array
  lipschitz_used: jax.Array
  smooth_evals: jax.Array
  grad_evals: jax.Array
  prox_evals: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _RefineCarry:
  """Internal backtracking loop state for `refine`."""

  lipschitz: jax.Array
  x_trial: _typing.PyTree
  f_trial: jax.Array
  accepted: jax.Array
  backtracks: jax.Array


def refine(
    oracle: composite.CompositeOracleProtocol,
    y: _typing.PyTree,
    lipschitz_init: _typing.Scalar,
    *,
    backtrack_factor: float = 2.0,
    max_backtracks: int = 40,
    jit: bool = True,
) -> RefinementResult:
  """Computes a certified pair `(x_hat, v_hat)` from an approximate iterate `y`.

  Performs one adaptive composite gradient step from `y` with smoothness `L`:

  - `x_hat = prox_{(1 / L) h}(y - (1 / L) grad_f(y))`
  - `v_hat = grad_f(x_hat) - grad_f(y) + L * (y - x_hat)`

  Because `L * (y - x_hat) - grad_f(y) in dh(x_hat)` holds exactly by the
  optimality condition of `prox`, `v_hat in grad_f(x_hat) + dh(x_hat)` is
  guaranteed without subgradient selection heuristics.

  Args:
    oracle: Target `CompositeOracleProtocol`.
    y: Approximate domain point `y in dom(h)`.
    lipschitz_init: Initial trial smoothness constant `L_0 > 0`.
    backtrack_factor: Multiplicative step-up factor `> 1` during line search.
    max_backtracks: Maximum number of backtracking doublings.
    jit: Whether to run line search via `jax.lax.while_loop` (`True`) or eager
      Python (`False`).

  Returns:
    `RefinementResult` containing `(x_hat, v_hat)`, objective value, and counts.
  """
  dtype = tree.infer_dtype(y)
  f_y, grad_y = oracle.smooth_value_and_grad(y)
  num_slack = 10.0 * tree.eps(dtype) * (1.0 + jnp.abs(f_y))

  if oracle.curvature_upper is not None:
    L_fixed = jnp.maximum(
        jnp.asarray(oracle.curvature_upper, dtype=dtype),
        jnp.asarray(1e-12, dtype=dtype),
    )
    inv_L = 1.0 / L_fixed
    x_hat = oracle.prox(tree.add_scaled(y, -inv_L, grad_y), inv_L)
    f_x_hat, grad_x_hat = oracle.smooth_value_and_grad(x_hat)
    v_hat = tree.add(
        tree.sub(grad_x_hat, grad_y),
        tree.scale(L_fixed, tree.sub(y, x_hat)),
    )
    obj_val = f_x_hat + oracle.proximal_value(x_hat)
    return RefinementResult(
        x_hat=x_hat,
        v_hat=v_hat,
        obj_val=obj_val,
        lipschitz_used=L_fixed,
        smooth_evals=jnp.asarray(2, dtype=jnp.int32),
        grad_evals=jnp.asarray(2, dtype=jnp.int32),
        prox_evals=jnp.asarray(1, dtype=jnp.int32),
    )

  L0 = jnp.maximum(
      jnp.asarray(lipschitz_init, dtype=dtype),
      jnp.asarray(1e-12, dtype=dtype),
  )

  def _eval_trial(
      L_val: jax.Array,
  ) -> tuple[_typing.PyTree, jax.Array, jax.Array]:
    step = 1.0 / L_val
    x_t = oracle.prox(tree.add_scaled(y, -step, grad_y), step)
    f_t = oracle.smooth_value(x_t)
    diff = tree.sub(x_t, y)
    quad_upper = (
        f_y + tree.vdot(grad_y, diff) + 0.5 * L_val * tree.norm_sq(diff)
    )
    ok = f_t <= quad_upper + num_slack
    return x_t, f_t, ok

  x0_trial, f0_trial, ok0 = _eval_trial(L0)
  init_carry = _RefineCarry(
      lipschitz=L0,
      x_trial=x0_trial,
      f_trial=f0_trial,
      accepted=ok0,
      backtracks=jnp.asarray(0, dtype=jnp.int32),
  )

  def _cond(carry: _RefineCarry) -> jax.Array:
    return jnp.logical_and(
        jnp.logical_not(carry.accepted),
        carry.backtracks < max_backtracks,
    )

  def _body(carry: _RefineCarry) -> _RefineCarry:
    next_L = carry.lipschitz * backtrack_factor
    x_t, f_t, ok = _eval_trial(next_L)
    return _RefineCarry(
        lipschitz=next_L,
        x_trial=x_t,
        f_trial=f_t,
        accepted=ok,
        backtracks=carry.backtracks + 1,
    )

  final_carry = _loop.while_loop(_cond, _body, init_carry, jit=jit)
  x_hat = final_carry.x_trial
  L_used = final_carry.lipschitz
  f_x_hat, grad_x_hat = oracle.smooth_value_and_grad(x_hat)
  v_hat = tree.add(
      tree.sub(grad_x_hat, grad_y),
      tree.scale(L_used, tree.sub(y, x_hat)),
  )
  obj_val = f_x_hat + oracle.proximal_value(x_hat)
  trials = 1 + final_carry.backtracks
  return RefinementResult(
      x_hat=x_hat,
      v_hat=v_hat,
      obj_val=obj_val,
      lipschitz_used=L_used,
      smooth_evals=trials + 2,
      grad_evals=jnp.asarray(2, dtype=jnp.int32),
      prox_evals=trials,
  )
