"""Sphinx configuration for the Proxax Experiments & Benchmarks site."""

from __future__ import annotations

project = "Proxax Experiments"
author = "Weiwei Kong"
copyright = "2026, Weiwei Kong"

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
]
myst_heading_anchors = 3

html_theme = "sphinx_book_theme"
pygments_dark_style = "monokai"
html_title = "Proxax Experiments"
html_logo = "_static/logo.png"
html_favicon = "_static/logo.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["sortable_tables.js"]

html_theme_options = {
    "repository_url": "https://github.com/wwkong/proxax",
    "repository_branch": "main",
    "path_to_docs": "docs-experiments",
    "use_repository_button": True,
    "use_source_button": True,
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "show_toc_level": 2,
    "show_navbar_depth": 2,
    "navigation_with_keys": False,
}
