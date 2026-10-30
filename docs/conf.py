# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add the parent directory to the path so we can import pollux
sys.path.insert(0, os.path.abspath('..'))

project = 'Pollux'
copyright = '2026, LSST DESC'
author = 'Cyrille Doux'
release = '0.1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_private_members = False
napoleon_include_special_members = False
napoleon_use_ivar = True

# Autodoc settings
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

# Autosummary: generate stub pages for documented modules
autosummary_generate = True
autosummary_generate_overwrite = False

# Mock imports for heavy dependencies
autodoc_mock_imports = [
    'astropy',
    'BaryonForge',
    'fitsio',
    'healpy',
    'healsparse',
    'matplotlib',
    'numba',
    'parallel_sort',
    'parsl',
    'pyccl',
    'pygio',
    'pymaster',
    'pynverse',
    'sacc',
    'scipy',
    'skyproj',
    'tqdm',
    'treecorr',
    'yaml',
]

# Intersphinx mapping for external libraries
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}
