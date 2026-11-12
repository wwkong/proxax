from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

import proxax


def test_environment_and_package_smoke() -> None:
  x = jnp.array([1.0, -2.0, 3.0])
  projected = optax.projections.projection_non_negative(x)
  assert jnp.allclose(projected, jnp.array([1.0, 0.0, 3.0]))
  assert proxax.__version__ == "0.0.2"
  assert len(jax.devices()) >= 1
