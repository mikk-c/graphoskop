"""
Sphinx configuration for graphoskop.
"""

import os
import sys
from pathlib import Path

# Make the package importable without installing it.
sys.path.insert(0, str(Path("..").resolve()))

# Mock heavy dependencies that are not needed to extract docstrings.
autodoc_mock_imports = [
    "torch",
    "torch_geometric",
    "numba",
    "numpy",
    "pandas",
    "scipy",
    "tqdm",
]

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath(os.path.join("..", "gui", "python")))   # worker of the interface

project = "Graphoskop"
extensions = ["sphinx.ext.autodoc"]
autodoc_member_order = "bysource"
html_theme = "sphinx_rtd_theme"
