from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import optax

from proxax import constraints
from proxax import optax_compat
from proxax import prox


def test_optax_proximal_step() -> None:
  lr = 0.2
  tx = optax.chain(
      optax.scale(-lr),
      optax_compat.proximal(prox.NonNegative(), step_size=lr),
  )
  params = {"w": jnp.array([0.1, 0.5])}
  state = tx.init(params)
  grads = {"w": jnp.array([2.0, -1.0])}
  updates, state = tx.update(grads, state, params)
  next_params = optax.apply_updates(params, updates)
  assert jnp.allclose(next_params["w"], jnp.array([0.0, 0.7]))


def test_optax_aipp_transform() -> None:
  def smooth_fn(x: jax.Array) -> jax.Array:
    return 0.5 * (2.0 * x[0] ** 2 - 0.5 * x[1] ** 2) - 0.4 * x[0] + 0.3 * x[1]

  tx = optax_compat.aipp(
      prox_op=prox.Box(lower=0.0, upper=1.0),
      opt_tol=1e-3,
      max_inner_iters=30,
  )
  params = jnp.array([-0.5, 1.5])
  state = tx.init(params)

  @jax.jit
  def step(
      p: jax.Array,
      st: optax_compat.AIPPTransformState,
  ) -> tuple[jax.Array, optax_compat.AIPPTransformState]:
    val, grads = jax.value_and_grad(smooth_fn)(p)
    updates, next_st = tx.update(grads, st, p, value=val, value_fn=smooth_fn)
    return optax.apply_updates(p, updates), next_st

  for _ in range(40):
    params, state = step(params, state)
    if bool(state.converged):
      break

  assert bool(state.converged)
  assert jnp.all((params >= -1e-6) & (params <= 1.0 + 1e-6))
  assert float(state.opt_residual) <= 1e-3


def test_optax_qp_aipp_transform() -> None:
  def smooth_fn(p: dict[str, jax.Array]) -> jax.Array:
    x = p["x"]
    return 0.5 * jnp.sum((x - jnp.array([0.6, -0.2])) ** 2)

  tx = optax_compat.qp_aipp(
      constraints={
          "eq": constraints.equality(
              lambda p: jnp.sum(p["x"]),
              rhs=jnp.asarray(0.5),
          ),
      },
      prox_op=prox.Box(lower=-1.0, upper=1.0),
      lower_curvature=0.0,
      penalty_init=5.0,
      penalty_factor=10.0,
      opt_tol=2e-3,
      feas_tol=2e-3,
      max_aipp_iters=10,
      max_inner_iters=40,
  )
  params = {"x": jnp.array([0.25, 0.25])}
  state = tx.init(params)

  @jax.jit
  def run_loop(
      p0: dict[str, jax.Array],
      st0: optax_compat.QPTransformState,
  ) -> tuple[dict[str, jax.Array], optax_compat.QPTransformState]:
    def cond_fn(
        carry: tuple[dict[str, jax.Array], optax_compat.QPTransformState],
    ) -> jax.Array:
      _, st = carry
      return jnp.logical_and(
          jnp.logical_not(st.converged),
          st.total_steps < 250,
      )

    def body_fn(
        carry: tuple[dict[str, jax.Array], optax_compat.QPTransformState],
    ) -> tuple[dict[str, jax.Array], optax_compat.QPTransformState]:
      p, st = carry
      val, grads = jax.value_and_grad(smooth_fn)(p)
      updates, next_st = tx.update(grads, st, p, value=val, value_fn=smooth_fn)
      return optax.apply_updates(p, updates), next_st

    return jax.lax.while_loop(cond_fn, body_fn, (p0, st0))

  params, state = run_loop(params, state)

  assert bool(state.converged)
  assert int(state.penalty_iter) >= 1
  assert float(state.opt_residual) <= 2e-3
  assert float(state.feas_residual) <= 2e-3


def test_optax_apd_transform() -> None:
  def smooth_fn(x: jax.Array) -> jax.Array:
    return 0.5 * (2.0 * x[0] ** 2 - 0.5 * x[1] ** 2) - 0.4 * x[0] + 0.3 * x[1]

  tx = optax_compat.apd(
      prox_op=prox.Box(lower=0.0, upper=1.0),
      opt_tol=1e-3,
      max_inner_iters=40,
  )
  params = jnp.array([-0.5, 1.5])
  state = tx.init(params)

  @jax.jit
  def run_loop(
      p0: jax.Array,
      st0: optax_compat.APDTransformState,
  ) -> tuple[jax.Array, optax_compat.APDTransformState]:
    def cond_fn(
        carry: tuple[jax.Array, optax_compat.APDTransformState],
    ) -> jax.Array:
      _, st = carry
      return jnp.logical_and(
          jnp.logical_not(st.converged),
          st.total_steps < 120,
      )

    def body_fn(
        carry: tuple[jax.Array, optax_compat.APDTransformState],
    ) -> tuple[jax.Array, optax_compat.APDTransformState]:
      p, st = carry
      val, grads = jax.value_and_grad(smooth_fn)(p)
      updates, next_st = tx.update(grads, st, p, value=val, value_fn=smooth_fn)
      return optax.apply_updates(p, updates), next_st

    return jax.lax.while_loop(cond_fn, body_fn, (p0, st0))

  params, state = run_loop(params, state)

  assert bool(state.converged)
  assert jnp.all((params >= -1e-6) & (params <= 1.0 + 1e-6))
  assert float(state.opt_residual) <= 1e-3


def test_optax_qp_apd_transform() -> None:
  def smooth_fn(p: dict[str, jax.Array]) -> jax.Array:
    x = p["x"]
    return 0.5 * jnp.sum((x - jnp.array([0.6, -0.2])) ** 2)

  tx = optax_compat.qp_apd(
      constraints={
          "eq": constraints.equality(
              lambda p: jnp.sum(p["x"]),
              rhs=jnp.asarray(0.5),
          ),
      },
      prox_op=prox.Box(lower=-1.0, upper=1.0),
      lower_curvature=0.0,
      penalty_init=5.0,
      penalty_factor=10.0,
      opt_tol=2e-3,
      feas_tol=2e-3,
      max_apd_iters=15,
      max_inner_iters=50,
  )
  params = {"x": jnp.array([0.25, 0.25])}
  state = tx.init(params)

  @jax.jit
  def run_loop(
      p0: dict[str, jax.Array],
      st0: optax_compat.QPAPDTransformState,
  ) -> tuple[dict[str, jax.Array], optax_compat.QPAPDTransformState]:
    def cond_fn(
        carry: tuple[dict[str, jax.Array], optax_compat.QPAPDTransformState],
    ) -> jax.Array:
      _, st = carry
      return jnp.logical_and(
          jnp.logical_not(st.converged),
          st.total_steps < 300,
      )

    def body_fn(
        carry: tuple[dict[str, jax.Array], optax_compat.QPAPDTransformState],
    ) -> tuple[dict[str, jax.Array], optax_compat.QPAPDTransformState]:
      p, st = carry
      val, grads = jax.value_and_grad(smooth_fn)(p)
      updates, next_st = tx.update(grads, st, p, value=val, value_fn=smooth_fn)
      return optax.apply_updates(p, updates), next_st

    return jax.lax.while_loop(cond_fn, body_fn, (p0, st0))

  params, state = run_loop(params, state)

  assert bool(state.converged)
  assert int(state.penalty_iter) >= 1
  assert float(state.opt_residual) <= 2e-3
  assert float(state.feas_residual) <= 2e-3


def test_optax_ialm_transform() -> None:
  def smooth_fn(p: dict[str, jax.Array]) -> jax.Array:
    x = p["x"]
    return 0.5 * jnp.sum((x - jnp.array([0.6, -0.2])) ** 2)

  tx = optax_compat.ialm(
      constraints={
          "eq": constraints.equality(
              lambda p: jnp.sum(p["x"]),
              rhs=jnp.asarray(0.5),
          ),
      },
      prox_op=prox.Box(lower=-1.0, upper=1.0),
      lower_curvature=0.0,
      dampen=0.2,
      dual_step=0.8,
      penalty_init=2.0,
      penalty_factor=3.0,
      opt_tol=2e-3,
      feas_tol=2e-3,
      max_apd_iters=15,
      max_inner_iters=50,
  )
  params = {"x": jnp.array([0.1, 0.1])}
  state = tx.init(params)

  @jax.jit
  def run_loop(
      p0: dict[str, jax.Array],
      st0: optax_compat.IALMTransformState,
  ) -> tuple[dict[str, jax.Array], optax_compat.IALMTransformState]:
    def cond_fn(
        carry: tuple[dict[str, jax.Array], optax_compat.IALMTransformState],
    ) -> jax.Array:
      _, st = carry
      return jnp.logical_and(
          jnp.logical_not(st.converged),
          st.total_steps < 300,
      )

    def body_fn(
        carry: tuple[dict[str, jax.Array], optax_compat.IALMTransformState],
    ) -> tuple[dict[str, jax.Array], optax_compat.IALMTransformState]:
      p, st = carry
      val, grads = jax.value_and_grad(smooth_fn)(p)
      updates, next_st = tx.update(grads, st, p, value=val, value_fn=smooth_fn)
      return optax.apply_updates(p, updates), next_st

    return jax.lax.while_loop(cond_fn, body_fn, (p0, st0))

  params, state = run_loop(params, state)

  assert bool(state.converged)
  assert int(state.alm_iter) >= 1
  assert float(state.opt_residual) <= 2e-3
  assert float(state.feas_residual) <= 2e-3


def test_optax_states_compose_trackers_and_subproblems() -> None:
  params = {"x": jnp.array([0.2, -0.1])}
  eq = {
      "eq": constraints.equality(
          lambda p: jnp.sum(p["x"]),
          rhs=jnp.asarray(0.5),
      ),
  }
  box_prox = prox.Box(lower=-1.0, upper=1.0)

  aipp_st = optax_compat.aipp(prox_op=box_prox).init(params)
  apd_st = optax_compat.apd(prox_op=box_prox).init(params)
  qp_aipp_st = optax_compat.qp_aipp(constraints=eq, prox_op=box_prox).init(
      params
  )
  qp_apd_st = optax_compat.qp_apd(constraints=eq, prox_op=box_prox).init(params)
  ialm_st = optax_compat.ialm(constraints=eq, prox_op=box_prox).init(params)

  assert {f.name for f in dataclasses.fields(aipp_st)} == {
      "tracker",
      "subproblem",
  }
  assert isinstance(aipp_st.tracker, optax_compat.OptaxTracker)
  assert isinstance(aipp_st.subproblem, optax_compat.ACGSubproblemCarry)
  assert float(aipp_st.opt_residual) == float(aipp_st.tracker.opt_residual)
  assert float(aipp_st.lipschitz) == float(aipp_st.subproblem.lipschitz)

  assert {f.name for f in dataclasses.fields(apd_st)} == {
      "tracker",
      "subproblem",
  }
  assert isinstance(apd_st.tracker, optax_compat.OptaxTracker)
  assert isinstance(apd_st.subproblem, optax_compat.PFAPDSubproblemCarry)
  assert float(apd_st.lower_curvature) == float(
      apd_st.subproblem.lower_curvature
  )

  for constrained_st in (qp_aipp_st, qp_apd_st, ialm_st):
    field_names = {f.name for f in dataclasses.fields(constrained_st)}
    assert "tracker" in field_names
    assert "subproblem" in field_names
    assert isinstance(
        constrained_st.tracker,
        optax_compat.ConstrainedOptaxTracker,
    )
    assert float(constrained_st.feas_residual) == float(
        constrained_st.tracker.feas_residual
    )
  assert int(qp_aipp_st.penalty_iter) == int(qp_aipp_st.tracker.stage_iter)
  assert int(qp_apd_st.penalty_iter) == int(qp_apd_st.tracker.stage_iter)
  assert int(ialm_st.alm_iter) == int(ialm_st.tracker.stage_iter)
