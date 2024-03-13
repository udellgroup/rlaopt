# -- Path setup --------------------------------------------------------------

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "SketchyOpts"

# -- General configuration ---------------------------------------------------

master_doc = "index"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.katex",
    "sphinx_autodoc_typehints",
    "sphinx_design",
]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
autosummary_generate = True
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

# -- LaTex configuration ---------------------------------------------------

import sphinxcontrib.katex as katex

latex_macros = r"""
  \def \R                {\mathbb{R}}
"""
# Translate LaTeX macros to KaTeX and add to options for HTML builder
katex_macros = katex.latex_defs_to_katex_macros(latex_macros)
katex_options = "{displayMode: true, fleqn: true, macros: {" + katex_macros + "}}"
# Add LaTeX macros for LATEX builder
latex_elements = {"preamble": latex_macros}

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_theme_options = {
    "use_repository_button": True,
    "repository_url": "https://github.com/udellgroup/sketchyopts",
    "navigation_with_keys": False,
}
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]
