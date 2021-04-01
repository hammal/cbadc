# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import warnings


warnings.filterwarnings("ignore", category=UserWarning,
                        message='Matplotlib is currently using agg, which is a'
                                ' non-GUI backend, so cannot show the figure.')

sys.path.insert(0, os.path.abspath("../../src"))


# -- Project information -----------------------------------------------------

project = "Control-Bounded A/D Conversion Toolbox"
copyright = "2021, Hampus Malmberg"
author = "Hampus Malmberg"

# The full version, including alpha/beta/rc tags
release = "0.0.1"
version = "0.0.1"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
          "sphinx.ext.autodoc",
          "sphinx.ext.autosummary",
          "sphinx.ext.mathjax", 
          "sphinx_rtd_theme", 
          "nbsphinx", 
          "sphinx.ext.napoleon",
          "sphinx.ext.intersphinx",
          "sphinx_gallery.gen_gallery"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]

# intershpinx
intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
                        'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None)}


# Sphinx-gallery
from sphinx_gallery.sorting import FileNameSortKey
sphinx_gallery_conf = {
     'examples_dirs': '../code_examples',   # path to your example scripts
     'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
     
     # directory where function/class granular galleries are stored
     'backreferences_dir'  : 'backreferences',
     # Modules for which function/class level galleries are created. In
     # this case sphinx_gallery and numpy in a tuple of strings.
     'doc_module' : ('cbadc.analog_system', 
          'cbadc.analog_signal', 
          'cbadc.digital_control', 
          'cbadc.digital_estimator', 
          'cbadc.simulator',
          'cbadc.utilities'),
     'line_numbers': True,
     'remove_config_comments': True,
     'within_subsection_order': FileNameSortKey,
}

# generate autosummary even if no references
autosummary_generate = True