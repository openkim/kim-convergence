import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

project = "kim-convergence"
version = "0.0.2"
author = "Yaser Afshar"
copyright = f"2021-{datetime.now().year}, Regents of the University of Minnesota"
rst_epilog = f".. |copyright| replace:: {copyright}"

try:
    import kim_convergence

    release = kim_convergence.__version__
    version = ".".join(release.split(".")[:2])
except ImportError:
    pass


def setup(app):
    from numpydoc import __version__

    # remove the relabeller from the event loop
    app.disconnect(app.events.listeners["doctree-read"][-1])


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
    "conf_py_path": "/docs/",
}
