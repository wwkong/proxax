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
