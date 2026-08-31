import re
import sys
from pathlib import Path

root = Path(__file__).parents[1]
sys.path.insert(0, str(root))
release = re.search(
    r'__version__ = "([^"]+)"',
    (root / "opforch" / "__init__.py").read_text(encoding="utf-8"),
).group(1)

project = "opforch"
copyright = "2024, Gustavo de Rosa"
author = "Gustavo de Rosa"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]
autosummary_generate = True
autodoc_mock_imports = ["numpy", "torch"]
exclude_patterns = ["_build"]
html_theme = "alabaster"
autodoc_default_options = {"members": True, "show-inheritance": True}
autodoc_member_order = "bysource"
