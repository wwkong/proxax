"""Sphinx configuration for proxax documentation."""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

project = "proxax"
author = "William Kong"
release = "0.0.2"
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
myst_enable_extensions = ["dollarmath", "amsmath"]
html_theme = "furo"
autodoc_typehints = "description"
