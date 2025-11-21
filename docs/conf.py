"""Configuration file for the Sphinx documentation builder.
"""

import os
import sys
from pathlib import Path

# -- Path setup --------------------------------------------------------------

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'rlaopt'
copyright = '2025, Stanford University'
author = 'Pratik Rathore, Parth Nobel, Zachary Frangella'


release = '0.1.0'
version = '0.1.0'

# -- General configuration ---------------------------------------------------


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'myst_parser',  
]

templates_path = ['_templates']

# List of patternsto ignore when looking for source files
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# The suffix(es) of source filenames.
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# --  HTML output -------------------------------------------------


html_theme = 'sphinx_book_theme'


html_theme_options = {
    'repository_url': 'https://github.com/udellgroup/rlaopt',
    'use_repository_button': True,
    'use_edit_page_button': True,
    'use_issues_button': True,
    'home_page_in_toc': True,
    'show_navbar_depth': 2,
    'show_toc_level': 2,
    'navigation_depth': 3,
}

html_title = 'rlaopt Documentation'
html_logo = ''  # Add path if available
html_favicon = '' 

# Add any paths that contain custom static files 
html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------

# -- Options for autodoc extension -------------------------------------------
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

autodoc_typehints = 'description'

# -- Options for autosummary extension ---------------------------------------
autosummary_generate = True
autosummary_imported_members = True

# -- Options for Napoleon extension ------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Options for intersphinx extension ---------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pydantic': ('https://docs.pydantic.dev/', None),
}

# -- Options for MathJax ----------------------------------------------------
mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'

# -- Options for MyST Parser ------------------------------------------------
myst_enable_extensions = [
    'colon_fence',
    'dollarmath',
    'amsmath',
]
