"""
Sphinx configuration for graphoskop.
"""

import os
import sys
from pathlib import Path

# Add the repository root to Python's import path.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath(os.path.join("..", "gui", "python")))   # worker of the interface

project = "Graphoskop"
extensions = ["sphinx.ext.autodoc"]
autodoc_member_order = "bysource"
html_theme = "sphinx_rtd_theme"
