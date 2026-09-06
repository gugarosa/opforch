# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Configure the generated OPForch API documentation.

"""

import re
import sys
from pathlib import Path

root = Path(__file__).parents[1]
sys.path.insert(0, str(root))
version_match = re.search(
    r'__version__ = "([^"]+)"',
    (root / "opforch" / "__init__.py").read_text(encoding="utf-8"),
)
if version_match is None:
    raise ValueError("`__version__` must be defined in the package initializer.")

release = version_match.group(1)

project = "opforch"
copyright = "2026, Gustavo de Rosa"
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
