# Configuration file for the Sphinx documentation builder.

import glob
import os
import shutil
import sys

# Add the parent directory to the path so we can import pollux
sys.path.insert(0, os.path.abspath('..'))

# notebooks of the repository, copied into the documentation sources (nbsphinx only reads those)
_here = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(_here, 'notebooks'), exist_ok=True)
for _nb in glob.glob(os.path.join(_here, '..', 'notebooks', '*.ipynb')):
    shutil.copy(_nb, os.path.join(_here, 'notebooks'))

project = 'Pollux'
copyright = '2022-2026, LSST Dark Energy Science Collaboration (DESC)'
author = 'Cyrille Doux'
release = '0.1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'nbsphinx',
]

# notebooks are shown as saved, not run (they need the simulations and outputs of a run)
nbsphinx_execute = 'never'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']


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
