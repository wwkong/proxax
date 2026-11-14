"""Focused regression tests for fixes from the codebase audit."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import proxax as px
from proxax import constraints
from proxax import implicit as implicit_mod
from proxax import plotting
from proxax import prox
from proxax.oracle import block as block_mod
from proxax.oracle import terms


def test_batch_solve_with_time_limit() -> None:
  """`batch_solve(..., time_limit=...)` works with `threads` backend."""
  q_batch = jnp.array([[1.0, -2.0], [-1.0, 0.5]])
  x0 = jnp.zeros(2)

  def prob_fn(q: jax.Array) -> px.CompositeProblem:
    return px.CompositeProblem(
        smooth=lambda x: 0.5 * jnp.sum(x**2) + jnp.dot(q, x),
        proximal=prox.Box(lower=-1.0, upper=1.0),
    )

  sol = px.batch_solve(
      prob_fn,
      x0,
      q_batch,
      method="aipp",
      time_limit=10.0,
      backend="threads",
  )
  assert sol.x.shape == (2, 2)
  assert jnp.allclose(sol.x, jnp.array([[-1.0, 1.0], [1.0, -0.5]]), atol=1e-4)


def test_zero_norm_reverse_mode_ad_no_nan() -> None:
  """Zero-norm inputs to `L2`, `L21`, and `SecondOrderCone` give finite AD."""
  z = jnp.zeros(3)
  g_l2 = jax.grad(lambda v: jnp.sum(prox.L2(weight=1.0).prox(v, 0.5) ** 2))(z)
  assert jnp.all(jnp.isfinite(g_l2))

  z_mat = jnp.zeros((2, 3))
  g_l21 = jax.grad(
      lambda v: jnp.sum(prox.L21(weight=1.0, axis=-1).prox(v, 0.5) ** 2)
  )(z_mat)
  assert jnp.all(jnp.isfinite(g_l21))

  soc = constraints.SecondOrderCone()
  g_soc = jax.grad(lambda v: jnp.sum(soc.project(v) ** 2))(z)
  assert jnp.all(jnp.isfinite(g_soc))


def test_exotic_prox_edge_cases() -> None:
  """`Ellipsoid`, `Spectrahedron`, `LogBarrier`, `NegEntropy` handle edges."""
  ell = prox.Ellipsoid(center=jnp.zeros(3), radii=jnp.array([1.0, 2.0, 3.0]))
  p_ell = ell.prox(jnp.array([3.0, 0.0, 0.0]), 1.0)
  assert jnp.all(jnp.isfinite(p_ell))

  spec = prox.Spectrahedron(scale=0.0)
  p_spec = spec.prox(jnp.eye(2), 1.0)
  assert jnp.allclose(p_spec, jnp.zeros((2, 2)))

  spec_batched = prox.Spectrahedron(scale=1.0)
  batch_mats = jnp.stack([jnp.eye(2), 2.0 * jnp.eye(2)])
  p_batch = spec_batched.prox(batch_mats, 1.0)
  assert p_batch.shape == (2, 2, 2)
  assert jnp.all(jnp.isfinite(p_batch))

  lb = prox.LogBarrier(weight=1.0)
  p_lb = lb.prox(jnp.array([-1e4, 1.0]), 0.5)
  assert jnp.all(p_lb > 0.0) and jnp.all(jnp.isfinite(p_lb))

  ne = prox.NegEntropy(weight=1.0)
  p_ne = ne.prox(jnp.array([1e3, -1e3]), 0.5)
  assert jnp.all(p_ne >= 0.0) and jnp.all(jnp.isfinite(p_ne))


def test_block_solvers_track_history() -> None:
  """`ADMM`, `DPADMM`, and `PAdaptADMM` support `track_history=True`."""
  blocks = (
      block_mod.BlockTerm(
          smooth=terms.SmoothTerm(fn=lambda x: 0.5 * jnp.sum((x - 1.0) ** 2)),
          proximal=prox.Zero(),
          linear_op=jnp.eye(2),
      ),
      block_mod.BlockTerm(
          smooth=terms.SmoothTerm(fn=lambda x: 0.5 * jnp.sum((x + 1.0) ** 2)),
          proximal=prox.Zero(),
          linear_op=-jnp.eye(2),
      ),
  )
  prob = block_mod.BlockProblem(blocks=blocks, rhs=jnp.zeros(2))
  x0 = (jnp.zeros(2), jnp.zeros(2))
  for method in ("admm", "dp_admm", "p_adapt_admm"):
    sol = px.solve(
        prob, x0, method=method, opt_tol=1e-3, feas_tol=1e-3, track_history=True
    )
    assert sol.history is not None
    assert len(sol.history.outer_iters) >= 1


def test_implicit_rejects_cg_on_saddle_point() -> None:
  """`implicit_solve` rejects `linear_solver='cg'` on constrained problems."""

  def prob_fn(b: jax.Array) -> px.CompositeProblem:
    return px.CompositeProblem(
        smooth=lambda x: 0.5 * jnp.sum(x**2),
        constraints=constraints.equality(jnp.ones((1, 2)), rhs=b),
    )

  with pytest.raises(ValueError, match="SPD"):
    jax.grad(
        lambda b: jnp.sum(
            implicit_mod.implicit_solve(
                prob_fn,
                jnp.zeros(2),
                b,
                linear_solver="cg",
            )
        )
    )(jnp.array([1.0]))


def test_animate_phase_trajectory_empty_raises() -> None:
  """`animate_phase_trajectory` raises `ValueError` on empty trajectories."""
  with pytest.raises(ValueError, match="empty"):
    plotting.animate_phase_trajectory({"a": ([], [])})


def test_adap_fista_nonfinite_backtrack_recovers() -> None:
  """`AdapFISTA` shrinks `lam` when a trial step hits `inf`."""
  from proxax.solvers.unconstrained import adap_fista

  def smooth(x: jax.Array) -> jax.Array:
    # Blows up to inf for ||x|| > 5, smooth quadratic inside.
    r2 = jnp.sum(x**2)
    return jnp.where(r2 > 25.0, jnp.inf, 0.5 * r2 + jnp.sum(x))

  prob = px.CompositeProblem(smooth=smooth, proximal=prox.Zero())
  sol = px.solve(
      prob,
      jnp.array([2.0, 2.0]),
      solver=adap_fista.AdapFISTA(opt_tol=1e-4, lipschitz_init=0.1),
  )
  assert sol.status_enum == px.SolveStatus.CONVERGED
  assert jnp.allclose(sol.x, -jnp.ones(2), atol=1e-3)


def test_iaipal_large_penalty_no_int32_overflow() -> None:
  """`IAIPAL` clamps `k0_period` cleanly under large penalty / tiny `A`."""
  from proxax.solvers.constrained import iaipal

  a_mat = 1e-8 * jnp.eye(2)
  prob = px.CompositeProblem(
      smooth=lambda x: 0.5 * jnp.sum(x**2),
      proximal=prox.Box(lower=-1.0, upper=1.0),
      constraints=constraints.equality(a_mat, rhs=jnp.zeros(2)),
  )
  sol = px.solve(
      prob,
      jnp.array([0.5, -0.5]),
      solver=iaipal.IAIPAL(
          opt_tol=1e-3,
          feas_tol=1e-3,
          penalty_init=1e4,
          max_outer_iters=3,
          max_inner_iters=20,
      ),
  )
  assert jnp.all(jnp.isfinite(sol.x))


def test_optax_compat_rejects_ls_fail_and_nonfinite() -> None:
  """`_subproblems` helpers reject `LINE_SEARCH_FAILED` and `NaN`s."""
  from proxax.optax_compat import _subproblems
  from proxax.solvers import base as solver_base
  from proxax.solvers.inner import acg
  from proxax.solvers.inner import pf_acg

  acg_ls_fail = acg.ACGState(
      x=jnp.zeros(2),
      y=jnp.zeros(2),
      x0=jnp.zeros(2),
      psi_x0=jnp.array(1.0),
      psi_y=jnp.array(0.0),
      u=jnp.zeros(2),
      eta=jnp.array(0.0),
      a_sum=jnp.array(1.0),
      lipschitz=jnp.array(1.0),
      iteration=jnp.array(1, dtype=jnp.int32),
      status=jnp.array(int(acg.ACGStatus.LINE_SEARCH_FAILED), dtype=jnp.int32),
      counts=solver_base.OracleCounts.zeros(),
  )
  assert bool(
      _subproblems._check_aipp_rejection(
          acg_ls_fail,
          center=jnp.ones(2),
          center_obj=jnp.array(1.0),
          phi_y=jnp.array(0.0),
          lam=jnp.array(1.0),
          inv_lam=jnp.array(1.0),
          sigma=0.5,
          use_adaptive=True,
      )
  )

  pf_ls_fail = pf_acg.PFACGState(
      x=jnp.zeros(2),
      y=jnp.zeros(2),
      y0=jnp.zeros(2),
      psi_y0=jnp.array(1.0),
      psi_y=jnp.array(0.0),
      r=jnp.zeros(2),
      a_sum=jnp.array(1.0),
      lipschitz=jnp.array(1.0),
      iteration=jnp.array(1, dtype=jnp.int32),
      status=jnp.array(
          int(pf_acg.PFACGStatus.LINE_SEARCH_FAILED), dtype=jnp.int32
      ),
      counts=solver_base.OracleCounts.zeros(),
  )
  rej, _, _ = _subproblems._evaluate_apd_subproblem(
      pf_ls_fail,
      center=jnp.ones(2),
      center_obj=jnp.array(1.0),
      phi_y=jnp.array(0.0),
      u_cand=jnp.zeros(2),
      v_cand=jnp.zeros(2),
      m_hat=jnp.array(1.0),
      theta=0.5,
      eff_rho=0.5,
      done_flag=jnp.array(False),
  )
  assert bool(rej)


def test_pdhg_final_iter_eval_every() -> None:
  """`PDHG` evaluates certificates on the final iter when `eval_every` > cap."""
  from proxax.oracle import minmax as minmax_mod
  from proxax.solvers.minmax import pdhg

  prob = minmax_mod.MinMaxProblem(
      coupling=lambda x, y: (
          jnp.dot(x, y) - 0.5 * jnp.sum(y**2) + 0.5 * jnp.sum(x**2)
      ),
  )
  sol = px.solve(
      prob,
      (jnp.ones(2), jnp.ones(2)),
      solver=pdhg.PDHG(opt_tol=1e-4, max_iters=3, eval_every=10),
  )
  assert jnp.isfinite(sol.stats.opt_residual)
  assert sol.stats.outer_iters == 3
