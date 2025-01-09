import os

# Project Information
project = 'rust_nurbs'
copyright = '2025, Matthew G. Lauer'
author = 'Matthew G. Lauer'

# Release Information
with open(os.path.join("..", "..", "Cargo.toml"), "r") as toml_file:
    lines = toml_file.readlines()
version = lines[2].split("=")[-1].strip().replace('"', '')
release = ".".join(version.split(".")[:-1])

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

# Static path
html_static_path = ['_static']

# Custom CSS file location
html_css_files = [
    'css/custom.css',
]

# Logo
html_logo = "_static/logo.png"

# Auto API (reading .pyi files)
autoapi_type = 'python'
autoapi_dirs = ['../..']
autoapi_file_patterns = ['rust*.pyi']
autoapi_ignore = ['*.rst', '*migrations*']
autoapi_add_toctree_entry = False
