from __future__ import annotations

import pathlib
import re
import tomllib

import jax
import jax.numpy as jnp
import optax

import proxax
from proxax import px


def test_environment_and_package_smoke() -> None:
  x = jnp.array([1.0, -2.0, 3.0])
  projected = optax.projections.projection_non_negative(x)
  assert jnp.allclose(projected, jnp.array([1.0, 0.0, 3.0]))
  assert proxax.__version__ == "0.2.3"
  assert px.__version__ == "0.2.3"
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


def test_version_sync() -> None:
  """Ensures package, citation, README, and docs versions stay in sync."""
  root = pathlib.Path(__file__).resolve().parent.parent
  expected = tomllib.loads(
      (root / "pyproject.toml").read_text(encoding="utf-8")
  )["project"]["version"]
  assert proxax.__version__ == expected
  assert px.__version__ == expected

  cff = (root / "CITATION.cff").read_text(encoding="utf-8")
  cff_match = re.search(r'^version:\s*"([^"]+)"', cff, flags=re.MULTILINE)
  assert cff_match is not None and cff_match.group(1) == expected

  for doc_rel in ("README.md", "docs/index.md"):
    text = (root / doc_rel).read_text(encoding="utf-8")
    bib_match = re.search(r"version\s*=\s*\{([^}]+)\}", text)
    assert bib_match is not None and bib_match.group(1) == expected
  assert hasattr(px.typing, "PyTree")
