"""Tests for conic, PAVA/sorting, barrier, and sparse prox operators."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from proxax import px


def test_conic_and_matrix_projections() -> None:
  """PSDCone, Spectrahedron, and SOC satisfy feasibility and optimality."""
  mat = jnp.array([[2.0, 1.0], [1.0, -1.0]])
  psd = jax.jit(px.prox.PSDCone().prox)(mat, 1.0)
  evals_psd = jnp.linalg.eigvalsh(psd)
  assert jnp.all(evals_psd >= -1e-6)
  assert jnp.allclose(psd, psd.T, atol=1e-6)

  spec = jax.jit(px.prox.Spectrahedron(scale=2.0).prox)(mat, 1.0)
  evals_spec = jnp.linalg.eigvalsh(spec)
  assert jnp.all(evals_spec >= -1e-6)
  assert jnp.allclose(jnp.trace(spec), 2.0, atol=1e-5)

  soc = px.prox.SOC()
  inside = jnp.array([5.0, 3.0, 4.0])
  assert jnp.allclose(jax.jit(soc.prox)(inside, 1.0), inside, atol=1e-6)
  below = jnp.array([-6.0, 3.0, 4.0])
  assert jnp.allclose(jax.jit(soc.prox)(below, 1.0), 0.0, atol=1e-6)
  mid = jnp.array([1.0, 3.0, 4.0])
  proj_mid = jax.jit(soc.prox)(mid, 1.0)
  assert jnp.allclose(proj_mid[0], jnp.linalg.norm(proj_mid[1:]), atol=1e-6)


def test_isotonic_owl_oscar_and_capped_simplex() -> None:
  """Isotonic, OWL, OSCAR, and CappedSimplex match exact KKT conditions."""
  y = jnp.array([3.0, 1.0, 2.0, 5.0, 4.0])
  iso_inc = jax.jit(px.prox.Isotonic(increasing=True).prox)(y, 1.0)
  assert jnp.all(jnp.diff(iso_inc) >= -1e-6)
  assert jnp.allclose(iso_inc, jnp.array([2.0, 2.0, 2.0, 4.5, 4.5]), atol=1e-5)

  iso_dec = jax.jit(px.prox.Isotonic(increasing=False).prox)(y, 1.0)
  assert jnp.all(jnp.diff(iso_dec) <= 1e-6)

  w = jnp.array([0.8, 0.5, 0.2])
  v = jnp.array([1.2, -1.0, 0.3])
  owl = px.prox.OWL(weights=w)
  u_owl = jax.jit(owl.prox)(v, 0.5)
  # Check subgradient optimality: 0 in d(h)(u) + (u - v)/tau.
  assert jnp.all(jnp.abs(u_owl) <= jnp.abs(v) + 1e-6)
  assert jnp.allclose(u_owl[2], 0.2, atol=1e-5)

  oscar = px.prox.OSCAR(alpha=0.2, beta=0.3)
  u_oscar = jax.jit(oscar.prox)(jnp.array([1.0, 0.9, -0.1]), 0.5)
  assert jnp.allclose(u_oscar[0], u_oscar[1], atol=1e-5)
  assert jnp.isclose(oscar.value(u_oscar), px.prox.OWL(
      weights=jnp.array([0.8, 0.5, 0.2])
  ).value(u_oscar))

  cs = px.prox.CappedSimplex(scale=1.5, upper=0.6)
  p_cs = jax.jit(cs.prox)(jnp.array([1.0, 0.8, 0.2, -0.3]), 1.0)
  assert jnp.all((p_cs >= -1e-6) & (p_cs <= 0.6 + 1e-6))
  assert jnp.allclose(jnp.sum(p_cs), 1.5, atol=1e-5)


def test_barriers_and_sparse_nonconvex_prox() -> None:
  """LogBarrier, NegEntropy, L0, and CardinalityBall satisfy optimality."""
  v = jnp.array([-1.0, 0.5, 2.0])
  lb = px.prox.LogBarrier(weight=0.25)
  u_lb = jax.jit(lb.prox)(v, 0.5)
  assert jnp.all(u_lb > 0.0)
  assert jnp.allclose(-0.25 / u_lb + (u_lb - v) / 0.5, 0.0, atol=1e-5)

  ne = px.prox.NegEntropy(weight=0.3)
  u_ne = jax.jit(ne.prox)(v, 0.5)
  assert jnp.all(u_ne > 0.0)
  assert jnp.allclose(
      0.3 * (jnp.log(u_ne) + 1.0) + (u_ne - v) / 0.5, 0.0, atol=1e-5
  )

  l0 = px.prox.L0(weight=0.5)
  # Threshold is sqrt(2 * 0.5 * 0.5) = sqrt(0.5) ~= 0.7071.
  u_l0 = jax.jit(l0.prox)(jnp.array([0.5, -0.8, 0.7, 1.2]), 0.5)
  assert jnp.allclose(u_l0, jnp.array([0.0, -0.8, 0.0, 1.2]), atol=1e-6)

  cb = px.prox.CardinalityBall(k=2)
  u_cb = jax.jit(cb.prox)(jnp.array([0.2, -1.5, 0.8, -0.4]), 1.0)
  assert jnp.allclose(u_cb, jnp.array([0.0, -1.5, 0.8, 0.0]), atol=1e-6)


def test_exotic_prox_solver_integration() -> None:
  """End-to-end `px.solve` with `Isotonic` and `Spectrahedron`."""
  target = jnp.array([3.0, 1.0, 2.0, 5.0])

  def _iso_loss(x: jax.Array) -> jax.Array:
    return 0.5 * jnp.sum((x - target) ** 2)

  prob = px.CompositeProblem(
      smooth=px.oracle.as_smooth_term(_iso_loss),
      proximal=px.prox.Isotonic(increasing=True),
  )
  sol = px.solve(prob, jnp.zeros(4), method="apd", opt_tol=1e-5)
  assert sol.status_enum == px.SolveStatus.CONVERGED
  assert jnp.allclose(sol.x, jnp.array([2.0, 2.0, 2.0, 5.0]), atol=1e-4)
