"""Sphinx configuration for Proxax documentation (JAX-style Book Theme)."""

from __future__ import annotations

import os
import pathlib
import tomllib

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_PYPROJECT = tomllib.loads(
    (pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml")
    .read_text(encoding="utf-8")
)

project = "Proxax"
author = "Weiwei Kong"
copyright = "2026, Weiwei Kong"
release = _PYPROJECT["project"]["version"]

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
]
myst_heading_anchors = 3

html_theme = "sphinx_book_theme"
html_title = "Proxax"
html_logo = "_static/logo.png"
html_favicon = "_static/logo.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["sortable_tables.js"]

html_theme_options = {
    "repository_url": "https://github.com/wwkong/proxax",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_source_button": True,
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "show_toc_level": 2,
    "show_navbar_depth": 2,
    "navigation_with_keys": False,
}

copybutton_prompt_text = (
    r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
)
copybutton_prompt_is_regexp = True

autodoc_typehints = "description"
autodoc_member_order = "bysource"
