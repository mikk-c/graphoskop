import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gui" / "python"))

# Load SciPy while PyTorch is not mocked.
import scipy.stats

if os.environ.get("READTHEDOCS") == "True":
    autodoc_mock_imports = [
        "torch",
        "torch_geometric",
    ]

project = "Graphoskop"
extensions = ["sphinx.ext.autodoc"]
autodoc_member_order = "bysource"
html_theme = "sphinx_rtd_theme"
