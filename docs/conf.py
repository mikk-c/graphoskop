"""
Sphinx configuration for graphoskop.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath(os.path.join("..", "gui", "python")))   # worker of the interface

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if os.environ.get("READTHEDOCS") == "True":
    autodoc_mock_imports = [
        "torch",
        "torch_geometric",
    ]

project = "Graphoskop"
extensions = ["sphinx.ext.autodoc"]
autodoc_member_order = "bysource"
html_theme = "sphinx_rtd_theme"
