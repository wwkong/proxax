from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

import proxax
from proxax import px


def test_environment_and_package_smoke() -> None:
  x = jnp.array([1.0, -2.0, 3.0])
  projected = optax.projections.projection_non_negative(x)
  assert jnp.allclose(projected, jnp.array([1.0, 0.0, 3.0]))
  assert proxax.__version__ == "0.1.3"
  assert px.__version__ == "0.1.3"
  assert len(jax.devices()) >= 1


def test_px_facade_matches_package_namespace() -> None:
  expected = set(px.__all__)
  assert set(proxax.__all__) == expected | {"px"}
  for name in px.__all__:
    assert getattr(px, name) is getattr(proxax, name)
  assert not hasattr(px, "APD")
  assert not hasattr(proxax, "APD")
  assert hasattr(px.solvers, "APD")
  assert hasattr(px.prox, "L1")
  assert hasattr(px.constraints, "equality")
  assert hasattr(px.oracle, "SmoothTerm")
  assert hasattr(px.typing, "PyTree")
