"""Tests for Hyperplane, Halfspace, Ellipsoid, Huber, and implicit_solve."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from proxax import px


def test_hyperplane_and_halfspace() -> None:
  """Hyperplane and Halfspace projections satisfy KKT optimality."""
  normal = jnp.array([1.0, 2.0, -1.0])
  offset = 3.0
  x = jnp.array([2.0, 3.0, 1.0])

  hp = px.prox.Hyperplane(normal=normal, offset=offset)
  p_hp = hp.prox(x, 1.0)
  assert jnp.allclose(jnp.dot(normal, p_hp), offset, atol=1e-6)
  # Residual x - p_hp must be parallel to normal
  diff = x - p_hp
  ratio = diff / normal
  assert jnp.allclose(ratio, ratio[0], atol=1e-6)

  hs = px.prox.Halfspace(normal=normal, offset=offset)
  p_hs_out = hs.prox(x, 1.0)
  assert jnp.allclose(p_hs_out, p_hp, atol=1e-6)
  x_in = jnp.array([0.0, 0.5, 0.0])
  assert jnp.allclose(hs.prox(x_in, 1.0), x_in, atol=1e-6)


def test_ellipsoid_projection() -> None:
  """Ellipsoid projection projects onto weighted ellipsoid boundary."""
  radii = jnp.array([2.0, 1.0, 0.5])
  center = jnp.array([1.0, -0.5, 0.25])
  ell = px.prox.Ellipsoid(radii=radii, center=center)

  x_in = jnp.array([1.2, -0.4, 0.3])
  assert jnp.allclose(ell.prox(x_in, 1.0), x_in, atol=1e-6)

  x_out = jnp.array([4.0, 2.0, -1.5])
  p_out = ell.prox(x_out, 1.0)
  val = jnp.sum(((p_out - center) / radii) ** 2)
  assert jnp.allclose(val, 1.0, atol=1e-6)
  # Normal condition: (x_out - p_out) proportional to (p_out - center) / radii^2
  grad = (p_out - center) / (radii ** 2)
  diff = x_out - p_out
  ratios = diff / grad
  assert jnp.allclose(ratios, ratios[0], atol=1e-5)


def test_huber_prox() -> None:
  """Huber prox matches optimality condition x + gw * dH(x) = v."""
  huber = px.prox.Huber(weight=1.5, delta=0.8)
  v = jnp.array([-2.5, -0.5, 0.0, 0.4, 3.0])
  step = 0.5
  x = huber.prox(v, step)
  # Check subgradient optimality: (v - x) / step == weight * grad_H(x)
  grad_h = jnp.where(
      jnp.abs(x) <= huber.delta,
      x,
      huber.delta * jnp.sign(x),
  )
  assert jnp.allclose(v - x, step * huber.weight * grad_h, atol=1e-6)


def test_constrained_implicit_solve_panoc_alm() -> None:
  """`px.implicit_solve` differentiates through `panoc_alm`."""

  def _prob(target_sum: jax.Array) -> px.CompositeProblem:
    def _quad(x: jax.Array) -> jax.Array:
      return 0.5 * jnp.sum((x - jnp.array([1.0, 2.0, 3.0])) ** 2)

    A = jnp.ones((1, 3))
    return px.CompositeProblem(
        smooth=px.oracle.as_smooth_term(_quad),
        proximal=px.prox.L1(weight=0.1),
        constraints=px.constraints.equality(A, jnp.atleast_1d(target_sum)),
    )

  def _loss(target_sum: jax.Array) -> jax.Array:
    x_hat = px.implicit_solve(
        _prob,
        jnp.zeros(3),
        target_sum,
        method="panoc_alm",
        opt_tol=1e-5,
        feas_tol=1e-5,
        linear_tol=1e-5,
    )
    return jnp.sum(x_hat ** 2)

  grad = jax.grad(_loss)(jnp.array(3.0))
  assert jnp.isfinite(grad)
  assert jnp.abs(grad) > 0.1
