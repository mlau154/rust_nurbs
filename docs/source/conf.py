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

# Custom PyPI logo file
html_js_files = [
   "pypi-icon.js"
]

# Icon links
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/mlau154/rust_nurbs",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome"
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/rust-nurbs/",
            "icon": "fa-custom fa-pypi",
            "type": "fontawesome"
        }
   ]
}

# Auto API (reading .pyi files)
autoapi_type = 'python'
autoapi_dirs = ['../..']
autoapi_file_patterns = ['rust*.pyi']
autoapi_ignore = ['*.rst', '*migrations*']
autoapi_add_toctree_entry = False
