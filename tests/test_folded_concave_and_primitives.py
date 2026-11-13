"""Tests for folded-concave decompositions, PSDSet, Fantope, L21, LInf, TV1D."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp
import pytest

from proxax import px


def test_psd_set_projection_and_constrained_solve() -> None:
  psd = px.constraints.PSDSet()
  mat = jnp.array([[2.0, 0.5], [0.5, -1.0]])
  proj = psd.project(mat)
  eigvals = jnp.linalg.eigvalsh(proj)
  assert jnp.allclose(proj, proj.T, atol=1e-6)
  assert jnp.all(eigvals >= -1e-6)

  # Batched projection
  batch = jnp.stack([mat, -jnp.eye(2)])
  proj_batch = psd.project(batch)
  assert proj_batch.shape == (2, 2, 2)
  assert jnp.allclose(proj_batch[1], jnp.zeros((2, 2)), atol=1e-6)

  # Constrained solve with PSDSet: min_X 0.5 ||X - mat||_F^2 s.t. X >= 0
  prob = px.CompositeProblem(
      smooth=lambda x: 0.5 * jnp.sum((x - mat) ** 2),
      constraints=px.constraints.in_set(px.constraints.identity, psd),
  )
  sol = px.solve(
      prob, jnp.zeros((2, 2)), method="qp_apd", opt_tol=1e-4, feas_tol=1e-4
  )
  assert jnp.allclose(sol.x, proj, atol=2e-3)


def test_fantope_projection_and_solve() -> None:
  fantope = px.prox.Fantope(k=2.0)
  mat = jnp.diag(jnp.array([3.0, 1.5, 0.2, -1.0]))
  proj = fantope.prox(mat, 1.0)
  eigvals = jnp.linalg.eigvalsh(proj)
  assert jnp.allclose(proj, proj.T, atol=1e-6)
  assert jnp.all(eigvals >= -1e-6)
  assert jnp.all(eigvals <= 1.0 + 1e-6)
  assert jnp.isclose(jnp.sum(eigvals), 2.0, atol=1e-5)

  # Solve max_X <A, X> (i.e. min -<A, X>) s.t. X in 2-Fantope
  a_mat = jnp.diag(jnp.array([4.0, 3.0, 1.0, 0.5]))
  prob = px.CompositeProblem(
      smooth=lambda x: -jnp.sum(a_mat * x),
      proximal=fantope,
  )
  sol = px.solve(prob, jnp.eye(4) * 0.5, method="apd", opt_tol=1e-4)
  expected = jnp.diag(jnp.array([1.0, 1.0, 0.0, 0.0]))
  assert jnp.allclose(sol.x, expected, atol=1e-3)


def test_l21_group_lasso() -> None:
  l21 = px.prox.L21(weight=0.5, axis=-1)
  x = jnp.array([[3.0, 4.0], [0.3, 0.4], [0.0, 0.0]])
  assert jnp.isclose(l21.value(x), 0.5 * (5.0 + 0.5), atol=1e-6)

  u = l21.prox(x, step_size=2.0)
  # Threshold = 2.0 * 0.5 = 1.0: first row shrinks from 5 to 4, others zeroed.
  assert jnp.allclose(u[0], jnp.array([2.4, 3.2]), atol=1e-6)
  assert jnp.allclose(u[1], jnp.zeros(2), atol=1e-6)
  assert jnp.allclose(u[2], jnp.zeros(2), atol=1e-6)


def test_linf_and_linf_ball() -> None:
  linf = px.prox.LInf(weight=2.0)
  x = jnp.array([3.0, -1.5, 0.5])
  assert jnp.isclose(linf.value(x), 6.0, atol=1e-6)

  step = 0.5
  u = linf.prox(x, step)
  # Moreau decomposition: x - u lies in L1 ball of radius step * weight = 1.0
  dual = x - u
  assert jnp.sum(jnp.abs(dual)) <= step * 2.0 + 1e-5
  # Check optimality: (x - u) in partial (step * LInf)(u)
  max_u = jnp.max(jnp.abs(u))
  assert jnp.isclose(jnp.dot(dual, u), step * 2.0 * max_u, atol=1e-5)

  ball = px.prox.LInfBall(radius=1.2)
  assert jnp.allclose(ball.prox(x, 1.0), jnp.array([1.2, -1.2, 0.5]), atol=1e-6)


def test_tv1d_prox_jit_and_vmap() -> None:
  tv = px.prox.TV1D(weight=0.5)
  x = jnp.array([0.0, 0.1, 2.0, 2.1, 1.9, -1.0])
  expected_val = 0.5 * jnp.sum(jnp.abs(jnp.diff(x)))
  assert jnp.isclose(tv.value(x), expected_val, atol=1e-6)

  # Verify TV1D prox against solving min_u 0.5 ||u - x||^2 + step * TV1D(u)
  step = 1.0
  u = jax.jit(tv.prox)(x, step)
  # First-order optimality via dual certificate:
  # For any perturbation v, 0.5||u+eps*v - x||^2 + step*TV(u+eps*v) >= opt
  obj_u = 0.5 * jnp.sum((u - x) ** 2) + step * tv.value(u)
  alt = jnp.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
  for direction in [jnp.ones_like(x), x - u, alt]:
    for eps in [-1e-3, 1e-3]:
      cand = u + eps * direction
      obj_cand = 0.5 * jnp.sum((cand - x) ** 2) + step * tv.value(cand)
      assert obj_cand >= obj_u - 1e-6

  # vmap compatibility
  batch = jnp.stack([x, 2.0 * x])
  u_batch = jax.jit(jax.vmap(lambda row: tv.prox(row, step)))(batch)
  assert u_batch.shape == (2, 6)
  assert jnp.allclose(u_batch[0], u, atol=1e-5)


@pytest.mark.parametrize(
    "kind", ["mcp", "scad", "logsum", "capped", "capped_smooth"]
)
def test_folded_concave_decompositions(kind: str) -> None:
  weight = 0.4
  exact_fn: typing.Callable[[jax.Array], jax.Array] | None = None
  if kind == "mcp":
    smooth, l1 = px.prox.mcp_decomposition(weight=weight, gamma=3.0)
    def exact_fn(t: jax.Array) -> jax.Array:
      abs_t = jnp.abs(t)
      return jnp.sum(jnp.where(
          abs_t <= 3.0 * weight,
          weight * abs_t - 0.5 * (t ** 2) / 3.0,
          0.5 * 3.0 * (weight ** 2),
      ))
  elif kind == "scad":
    smooth, l1 = px.prox.scad_decomposition(weight=weight, a=3.7)
    def exact_fn(t: jax.Array) -> jax.Array:
      abs_t = jnp.abs(t)
      return jnp.sum(jnp.where(
          abs_t <= weight,
          weight * abs_t,
          jnp.where(
              abs_t <= 3.7 * weight,
              (2 * 3.7 * weight * abs_t - abs_t ** 2 - weight ** 2) / (2 * 2.7),
              0.5 * (3.7 + 1.0) * (weight ** 2),
          ),
      ))
  elif kind == "logsum":
    smooth, l1 = px.prox.logsum_decomposition(weight=weight, eps=0.2)
    def exact_fn(t: jax.Array) -> jax.Array:
      return jnp.sum(weight * 0.2 * jnp.log1p(jnp.abs(t) / 0.2))
  elif kind == "capped":
    smooth, l1 = px.prox.capped_l1_decomposition(weight=weight, theta=0.5)
    def exact_fn(t: jax.Array) -> jax.Array:
      return jnp.sum(weight * jnp.minimum(jnp.abs(t), 0.5))
  else:
    smooth, l1 = px.prox.capped_l1_decomposition(
        weight=weight, theta=0.5, smoothing=0.1
    )

  x = jnp.array([-2.0, -0.3, 0.0, 0.25, 0.8, 3.0])
  val, grad = smooth.value_and_grad(x)
  assert jnp.allclose(grad[2], 0.0, atol=1e-6)
  if exact_fn is not None:
    assert jnp.isclose(val + l1.value(x), exact_fn(x), atol=1e-5)

  # Check autodiff agreement away from kinks
  ad_val, ad_grad = jax.value_and_grad(smooth.value)(x)
  assert jnp.isclose(val, ad_val, atol=1e-6)
  assert jnp.allclose(grad, ad_grad, atol=1e-5)

  # Solve a sparse regression problem with the decomposed penalty
  target = jnp.array([2.0, 0.0, -1.5, 0.0])
  prob = px.CompositeProblem(
      smooth=lambda z: 0.5 * jnp.sum((z - target) ** 2) + smooth.value(z),
      proximal=l1,
  )
  sol = px.solve(prob, jnp.zeros(4), method="apd", opt_tol=1e-4)
  assert sol.status == px.SolveStatus.CONVERGED
  assert jnp.abs(sol.x[1]) < 1e-3
  assert jnp.abs(sol.x[3]) < 1e-3
