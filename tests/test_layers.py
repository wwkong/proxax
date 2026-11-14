"""Tests for `px.layers` differentiable optimization layers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsparse

from proxax import px


def test_simplex_and_capped_simplex_layers() -> None:
  """`SimplexLayer` and `CappedSimplexLayer` support `vmap` and `grad`."""
  simplex = px.layers.SimplexLayer(scale=1.0)
  capped = px.layers.CappedSimplexLayer(scale=2.0, upper=1.0)

  scores = jnp.array([[2.0, 0.5, -1.0, 0.1], [0.1, 3.0, 1.5, -0.5]])
  out_s = jax.jit(simplex.batched)(scores)
  out_c = jax.jit(capped.batched)(scores)
  assert jnp.allclose(jnp.sum(out_s, axis=-1), 1.0, atol=1e-5)
  assert jnp.allclose(jnp.sum(out_c, axis=-1), 2.0, atol=1e-5)
  assert jnp.all(out_c <= 1.0 + 1e-5)

  def _loss(s: jax.Array) -> jax.Array:
    return jnp.sum(capped.batched(s) ** 2)

  grad = jax.grad(_loss)(scores)
  assert grad.shape == scores.shape
  assert jnp.all(jnp.isfinite(grad))


def test_box_and_qp_layers() -> None:
  """`BoxLayer` and `QPLayer` differentiate cleanly w.r.t. inputs."""
  box = px.layers.BoxLayer(lower=-0.5, upper=0.5)
  s = jnp.array([-1.0, 0.2, 1.0])
  out_b = box(s)
  assert jnp.allclose(out_b, jnp.array([-0.5, 0.2, 0.5]), atol=1e-5)

  Q = jnp.eye(3)
  qp = px.layers.QPLayer(Q=Q, proximal=px.prox.NonNegative())
  c = jnp.array([-1.0, -2.0, -0.5])
  A = jnp.ones((1, 3))
  b = jnp.array([1.0])

  def _qp_loss(b_val: jax.Array) -> jax.Array:
    x_hat = qp(c, A, b_val)
    return jnp.sum(x_hat**2)

  g_b = jax.grad(_qp_loss)(b)
  assert jnp.isfinite(g_b[0])
  assert jnp.abs(g_b[0]) > 0.1


def test_qp_layer_bcoo() -> None:
  """`QPLayer` supports `BCOO` `Q` and `A` matrices with implicit gradients."""
  q_dense = jnp.diag(jnp.array([2.0, 3.0, 4.0, 5.0]))
  a_dense = jnp.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
  q_bcoo = jsparse.BCOO.fromdense(q_dense)
  a_bcoo = jsparse.BCOO.fromdense(a_dense)
  c = jnp.array([-1.0, -2.0, -1.0, -2.0])
  b = jnp.array([1.0, 1.0])

  qp_dense = px.layers.QPLayer(Q=q_dense, opt_tol=1e-6, feas_tol=1e-6)
  qp_sparse = px.layers.QPLayer(Q=q_bcoo, opt_tol=1e-6, feas_tol=1e-6)

  gc_d, gb_d = jax.grad(
      lambda c_, b_: 0.5 * jnp.sum(qp_dense(c_, a_dense, b_) ** 2),
      argnums=(0, 1),
  )(c, b)
  gc_s, gb_s = jax.grad(
      lambda c_, b_: 0.5 * jnp.sum(qp_sparse(c_, a_bcoo, b_) ** 2),
      argnums=(0, 1),
  )(c, b)
  assert jnp.allclose(gc_d, gc_s, atol=1e-4)
  assert jnp.allclose(gb_d, gb_s, atol=1e-4)

  ga = jax.grad(
      lambda a_: 0.5 * jnp.sum(qp_sparse(c, a_, b) ** 2),
      allow_int=True,
  )(a_bcoo)
  assert jnp.all(jnp.isfinite(ga.data))
