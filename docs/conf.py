# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.

import os
import sys

import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath("."))


# -- Project information -----------------------------------------------------

project = "opforch"
copyright = "2024, Gustavo de Rosa"
author = "Gustavo de Rosa"

# The short X.Y version
version = "2.0.0"

# The full version, including alpha/beta/rc tags
release = "2.0.0"


# -- General configuration ---------------------------------------------------

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "autoapi.extension"]

autoapi_dirs = ["../opforch"]
autoapi_generate_api_docs = False

templates_path = ["_templates"]

source_suffix = ".rst"

master_doc = "index"

language = "en"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

pygments_style = None


# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"

html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_theme_options = {
    "collapse_navigation": False,
    "display_version": True,
    "logo_only": True,
}


# -- Options for HTMLHelp output ---------------------------------------------

htmlhelp_basename = "opforch_doc"


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {}

latex_documents = [
    (master_doc, "opforch.tex", "OPForch Documentation", "Gustavo de Rosa", "manual"),
]


# -- Options for manual page output ------------------------------------------

man_pages = [(master_doc, "opforch", "OPForch Documentation", [author], 1)]


# -- Options for Texinfo output ----------------------------------------------

texinfo_documents = [
    (
        master_doc,
        "opforch",
        "OPForch Documentation",
        author,
        "opforch",
        "PyTorch-Inspired Optimum-Path Forest Classifier.",
        "Miscellaneous",
    ),
]


# -- Options for Epub output -------------------------------------------------

epub_title = project

epub_exclude_files = ["search.html"]


# -- Extension configuration -------------------------------------------------
autodoc_default_options = {"exclude-members": "__weakref__"}

autodoc_member_order = "bysource"
