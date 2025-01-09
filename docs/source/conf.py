
# Project Information
project = 'rust_nurbs'
copyright = '2025, Matthew G. Lauer'
author = 'Matthew G. Lauer'

# Release Information
release = '0.1'
version = '0.1.5'

# Sphinx Extensions
extensions = [
    'sphinx.ext.autodoc',
    'autoapi.extension',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosectionlabel',
    'sphinx_design',
]

# Theme
html_theme = 'pydata_sphinx_theme'

# Templates Path
templates_path = ['_templates']

# Auto API (reading .pyi files)
autoapi_type = 'python'
autoapi_dirs = ['../..']
autoapi_file_patterns = ['rust*.pyi']
autoapi_ignore = ['*.rst', '*migrations*']
autoapi_add_toctree_entry = False
