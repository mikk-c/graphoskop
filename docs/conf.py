"""
Sphinx configuration for graphoskop.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "Graphoskop"
extensions = ["sphinx.ext.autodoc"]
autodoc_member_order = "bysource"
html_theme = "sphinx_rtd_theme"
