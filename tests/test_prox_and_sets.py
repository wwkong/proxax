from __future__ import annotations

import jax
import jax.numpy as jnp

from proxax import constraints
from proxax import prox
from proxax import tree


def test_prox_operators_optimality_conditions() -> None:
  x = jnp.array([-2.5, -0.3, 0.0, 0.4, 3.1])
  step = 0.5

  # L1 optimality: (x - p) / step in weight * d||p||_1
  l1 = prox.L1(0.8)
  p_l1 = l1.prox(x, step)
  subgrad = (x - p_l1) / step
  for p_i, g_i in zip(p_l1.tolist(), subgrad.tolist(), strict=True):
    if abs(p_i) > 1e-6:
      assert abs(g_i - 0.8 * (1.0 if p_i > 0 else -1.0)) < 1e-5
    else:
      assert abs(g_i) <= 0.8 + 1e-5

  # Shifted proximal calculus matches direct minimization of quadratic + L1
  center = jnp.array([1.0, -1.0, 0.5, 2.0, -0.5])
  reg = jnp.asarray(1.5)
  shifted = prox.Shifted(base=l1, center=center, reg=reg)
  p_shift = shifted.prox(x, step)
  # Check subgradient of base at p_shift:
  # 0 in d(base)(p) + reg * (p - center) + (p - x) / step
  eff_subgrad = (x - p_shift) / step - reg * (p_shift - center)
  for p_i, g_i in zip(p_shift.tolist(), eff_subgrad.tolist(), strict=True):
    if abs(p_i) > 1e-6:
      assert abs(g_i - 0.8 * (1.0 if p_i > 0 else -1.0)) < 1e-5
    else:
      assert abs(g_i) <= 0.8 + 1e-5


def test_indicators_and_multi_prox_on_pytrees() -> None:
  params = {
      "kernel": jnp.array([[-2.0, 0.2], [1.5, -0.1]]),
      "scale": jnp.array([-1.0, 0.5, 2.0]),
      "probs": jnp.array([0.2, -0.5, 1.8]),
  }
  mp = prox.MultiProx({
      "kernel": prox.L1(0.5),
      "scale": prox.Box(lower=0.0, upper=1.0),
      "probs": prox.Simplex(1.0),
  })
  dom_x = mp.domain_project(params)
  assert jnp.all(dom_x["scale"] >= 0.0) and jnp.all(dom_x["scale"] <= 1.0)
  assert jnp.all(dom_x["probs"] >= -1e-6)
  assert jnp.isclose(jnp.sum(dom_x["probs"]), 1.0, atol=1e-5)

  prox_x = jax.jit(lambda p: mp.prox(p, 0.4))(params)
  assert jnp.allclose(prox_x["kernel"], jnp.array([[-1.8, 0.0], [1.3, 0.0]]))
  assert jnp.allclose(prox_x["scale"], jnp.array([0.0, 0.5, 1.0]))
  assert jnp.isclose(jnp.sum(prox_x["probs"]), 1.0, atol=1e-5)


def test_convex_sets_and_soc_projection() -> None:
  # Inside SOC
  y_in = jnp.array([5.0, 3.0, 4.0])
  soc = constraints.SecondOrderCone()
  assert jnp.allclose(soc.project(y_in), y_in)

  # Below negative polar cone -> projects to 0
  y_zero = jnp.array([-6.0, 3.0, 4.0])
  assert jnp.allclose(soc.project(y_zero), jnp.zeros_like(y_zero))

  # Boundary projection: ||u_proj||_2 == t_proj
  y_mid = jnp.array([1.0, 3.0, 4.0])  # ||u|| = 5 > 1
  proj_mid = soc.project(y_mid)
  assert jnp.isclose(proj_mid[0], 3.0)
  assert jnp.isclose(jnp.linalg.norm(proj_mid[1:]), proj_mid[0], atol=1e-6)
  assert tree.l2_norm(y_mid - proj_mid) > 0.0
