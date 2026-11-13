"""Syncs the package version across pyproject, px, CITATION.cff, and docs."""

from __future__ import annotations

import datetime
import pathlib
import re
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent


def bump_version(new_version: str) -> None:
  """Updates all versioned files in the repository to `new_version`."""
  today = datetime.date.today().isoformat()
  replacements: list[tuple[str, str, str]] = [
      (
          "pyproject.toml",
          r'(?m)^version = "[^"]+"',
          f'version = "{new_version}"',
      ),
      (
          "src/proxax/px.py",
          r'(?m)^__version__ = "[^"]+"',
          f'__version__ = "{new_version}"',
      ),
      (
          "CITATION.cff",
          r'(?m)^version: "[^"]+"',
          f'version: "{new_version}"',
      ),
      (
          "CITATION.cff",
          r'(?m)^date-released: "[^"]+"',
          f'date-released: "{today}"',
      ),
      (
          "README.md",
          r"version = \{[^}]+\}",
          f"version = {{{new_version}}}",
      ),
      (
          "docs/index.md",
          r"version = \{[^}]+\}",
          f"version = {{{new_version}}}",
      ),
      (
          "tests/test_smoke.py",
          r'(?m)^(\s+assert (?:proxax|px)\.__version__ == )"[^"]+"',
          rf'\1"{new_version}"',
      ),
  ]
  for rel_path, pattern, repl in replacements:
    path = _ROOT / rel_path
    text = path.read_text(encoding="utf-8")
    updated, count = re.subn(pattern, repl, text)
    if count == 0:
      raise RuntimeError(f"Pattern {pattern!r} did not match in {rel_path}")
    path.write_text(updated, encoding="utf-8")
  print(f"Synced version to {new_version}")


if __name__ == "__main__":
  if len(sys.argv) != 2:
    raise SystemExit("Usage: python scripts/bump_version.py <version>")
  bump_version(sys.argv[1])
