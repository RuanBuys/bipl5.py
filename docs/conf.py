# Sphinx configuration for the bipl5 documentation.
#
# Tutorials are MyST-Markdown notebooks executed at build time, so every
# page embeds live interactive biplots. The API reference is generated
# with autodoc from the package's RST-flavoured docstrings.

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "bipl5"
copyright = "2026, Ruan Buys"
author = "Ruan Buys"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "make.bat", "Makefile"]

# ── MyST / notebook execution ───────────────────────────────────────────────
myst_enable_extensions = ["colon_fence", "dollarmath"]
nb_execution_mode = "auto"          # text-based notebooks are always executed
nb_execution_timeout = 600
nb_execution_raise_on_error = True

# ── autodoc ─────────────────────────────────────────────────────────────────
autodoc_member_order = "bysource"
autodoc_typehints = "description"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}

# ── HTML output ─────────────────────────────────────────────────────────────
html_theme = "furo"
html_title = "bipl5"
html_theme_options = {
    "source_repository": "https://github.com/RuanBuys/bipl5.py",
    "source_branch": "main",
    "source_directory": "docs/",
}
