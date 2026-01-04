import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

project = "kim-convergence"
author = "Yaser Afshar"
copyright = f"2021-{datetime.now().year}, Regents of the University of Minnesota"
rst_epilog = f".. |copyright| replace:: {copyright}"

try:
    import kim_convergence

    release = kim_convergence.__version__
    version = ".".join(release.split(".")[:2])
except ImportError:
    import warnings

    warnings.warn(
        "Could not import kim_convergence; using fallback version", stacklevel=2
    )
    version = "0.0.3"


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinxcontrib.mermaid",
    "sphinx_design",
    "myst_parser",
]

language = "en"
source_suffix = ".rst"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
numpydoc_show_class_members = True

myst_enable_extensions = [
    "html_image",
    "html_admonition",
    "colon_fence",
    "deflist",
    "tasklist",
    "fieldlist",
    "replacements",
    "substitution",
    "linkify",
]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static", "files"]

# conf.py
html_context = {
    "display_github": True,
    "github_user": "openkim",
    "github_repo": "kim-convergence",
    "github_version": "main",
    "conf_py_path": "/doc/",
}
