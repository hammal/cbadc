# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import warnings


warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Matplotlib is currently using agg, which is a"
    " non-GUI backend, so cannot show the figure.",
)

# -- Project information -----------------------------------------------------

project = "cbadc"
copyright = "2025, Hampus Malmberg"
author = "Hampus Malmberg"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx_rtd_theme",
    "myst_nb",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.ifconfig",
    "sphinx.ext.githubpages",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.graphviz",
    "sphinx.ext.extlinks",
    "sphinx_gallery.gen_gallery",
]

# -- Sphinx-Gallery (executable tutorial scripts) -----------------------------
# ``docs/tutorials/plot_*.py`` are run at build time and rendered under
# ``tutorials/``; a script that raises fails the build, so the tutorials cannot
# drift from the API. Each script also runs standalone with ``python``.
sphinx_gallery_conf = {
    "examples_dirs": "../tutorials",
    "gallery_dirs": "tutorials",
    "filename_pattern": r"/plot_",
    "within_subsection_order": "FileNameSortKey",  # plot_00 (the primary tutorial) first
    "download_all_examples": False,
    "remove_config_comments": True,
    "matplotlib_animations": False,
}

# -- MyST-NB (executable markdown notebooks) ---------------------------------
# Notebooks under ``tutorials/`` are executed at build time; a failing cell
# fails the docs build, so they cannot silently drift from the API.
# "cache": execute once and reuse stored outputs on later builds (keyed on the
# code cells). The cache lives in docs/build/.jupyter_cache and survives
# clearing docs/build/html.
nb_execution_mode = "cache"
nb_execution_raise_on_error = True
nb_execution_timeout = 1800

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
# Sphinx-Gallery writes a downloadable notebook and script next to each rendered
# tutorial; without this, myst-nb picks the notebook up as a second source.
exclude_patterns = ["tutorials/*.ipynb", "tutorials/*.py", "_build"]


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
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
}

myst_enable_extensions = [
    "dollarmath",  # Support for LaTeX math
    "colon_fence",  # Support for colon-delimited fences
    "substitution",  # Support for variable substitution
]

# generate autosummary even if no references
autosummary_generate = True
autosectionlabel_prefix_document = True

# Show the ``.. todo::`` placeholders of the primary tutorial in the rendered docs.
todo_include_todos = True
