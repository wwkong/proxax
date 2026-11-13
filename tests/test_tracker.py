"""Tests for composable solver trackers, counters, and property mixins."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from proxax import px
from proxax.solvers import base


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _MockUnconstrainedState(base.SolverTrackerMixin):
  x: jax.Array
  tracker: base.SolverTracker


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _MockConstrainedState(base.ConstrainedTrackerMixin):
  x: jax.Array
  tracker: base.ConstrainedTracker


def test_solver_tracker_pytree_and_properties() -> None:
  x0 = jnp.array([2.0, -1.0])
  v0 = jnp.array([2.0, -1.0])
  tr0 = px.solvers.SolverTracker.from_init(
      x0,
      v0,
      jnp.linalg.norm(v0),
      jnp.float32(2.5),
      opt_tol=1e-3,
      smooth_evals=1,
      grad_evals=1,
      prox_evals=1,
  )
  st0 = _MockUnconstrainedState(x=x0, tracker=tr0)
  assert st0.status_enum == px.SolveStatus.RUNNING
  assert int(st0.smooth_evals) == 1

  def _step(st: _MockUnconstrainedState) -> _MockUnconstrainedState:
    x_next = 0.2 * st.x
    v_next = x_next
    res_next = jnp.linalg.norm(v_next)
    obj_next = 0.5 * jnp.sum(x_next**2)
    next_tr = st.tracker.update_incumbent(
        x_next, v_next, res_next, obj_next
    ).advance(
        inner_inc=1,
        smooth_inc=1,
        grad_inc=1,
        prox_inc=1,
        max_iters=20,
    )
    return _MockUnconstrainedState(x=x_next, tracker=next_tr)

  fin = base.run_solver_loop(_step, st0, jit=True)
  assert fin.status_enum == px.SolveStatus.CONVERGED
  assert float(fin.best_opt_res) <= 1e-3
  sol = fin.to_solution()
  assert sol.dual is None
  assert int(sol.stats.outer_iters) == int(fin.outer_iters)
  assert int(sol.stats.smooth_evals) == int(fin.smooth_evals)


def test_constrained_tracker_kkt_incumbent_and_solution() -> None:
  x0 = jnp.array([1.0, 1.0])
  p0 = jnp.array([0.0])
  v0 = jnp.array([1.0, 1.0])
  tr0 = px.solvers.ConstrainedTracker.from_init(
      x0,
      p0,
      v0,
      jnp.float32(1.0),
      jnp.float32(1.0),
      jnp.float32(1.0),
      opt_tol=1e-3,
      feas_tol=1e-3,
      smooth_evals=1,
  )
  st0 = _MockConstrainedState(x=x0, tracker=tr0)
  assert st0.status_enum == px.SolveStatus.RUNNING

  tr1 = st0.tracker.update_incumbent(
      jnp.array([0.5, 0.5]),
      jnp.array([0.2]),
      jnp.array([1e-4, 0.0]),
      jnp.float32(1e-4),
      jnp.float32(2e-4),
      jnp.float32(0.25),
  ).advance(inner_inc=3, smooth_inc=4, grad_inc=3, prox_inc=3, max_iters=10)
  st1 = _MockConstrainedState(x=jnp.array([0.5, 0.5]), tracker=tr1)
  assert st1.status_enum == px.SolveStatus.CONVERGED
  assert float(st1.best_feas_res) <= 1e-3
  sol = st1.to_solution()
  assert sol.dual is not None
  assert jnp.allclose(sol.dual, jnp.array([0.2]))


def test_solver_states_compose_trackers_and_expose_properties() -> None:
  unconstrained_fields = {
      f.name for f in dataclasses.fields(px.solvers.APDState)
  }
  assert "tracker" in unconstrained_fields
  assert "best_x" not in unconstrained_fields

  constrained_fields = {
      f.name for f in dataclasses.fields(px.solvers.IALMState)
  }
  assert "tracker" in constrained_fields
  assert "best_dual" not in constrained_fields

  prob = px.CompositeProblem(
      smooth=lambda x: 0.5 * jnp.sum((x - 1.0) ** 2),
      proximal=px.prox.L1(0.1),
  )
  apd_st = px.solvers.APD(opt_tol=1e-5).solve_oracle(prob.oracle, jnp.zeros(3))
  assert isinstance(apd_st.tracker, px.solvers.SolverTracker)
  assert isinstance(apd_st.counts, px.solvers.OracleCounts)
  assert jnp.array_equal(apd_st.best_x, apd_st.tracker.best_x)
  assert int(apd_st.smooth_evals) == int(apd_st.tracker.counts.smooth_evals)

  c_prob = px.CompositeProblem(
      smooth=lambda x: 0.5 * jnp.sum((x - 2.0) ** 2),
      constraints=px.constraints.equality(jnp.ones((1, 2)), jnp.array([1.0])),
  )
  ialm_st = px.solvers.IALM(opt_tol=1e-5, feas_tol=1e-5).solve_state(
      c_prob, jnp.zeros(2)
  )
  assert isinstance(ialm_st.tracker, px.solvers.ConstrainedTracker)
  assert jnp.array_equal(ialm_st.best_dual, ialm_st.tracker.best_dual)
  assert float(ialm_st.best_feas_res) == float(ialm_st.tracker.best_feas_res)
