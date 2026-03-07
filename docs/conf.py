import os
import sys

project   = "pyregg"
copyright = "2025, Sarat Moka"
author    = "Sarat Moka"
release   = "0.2.8"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
]

# Mock heavy/optional imports so autodoc works without a full runtime install
autodoc_mock_imports = ["numba", "IPython"]

# NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring  = True
napoleon_use_param        = True
napoleon_use_rtype        = False

# Preserve source order (naive_mc → conditional_mc → importance_sampling)
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members":        True,
    "undoc-members":  False,
    "show-inheritance": False,
}

autosummary_generate = True

html_theme = "sphinx_rtd_theme"
html_static_path = []

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
