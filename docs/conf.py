"""Sphinx configuration for the sysone documentation.

Build locally with `uv run --group docs sphinx-build -W docs docs/_build/html`.
"""

import shutil
from pathlib import Path

import sysone

ROOT = Path(__file__).resolve().parent.parent

project = "sysone"
author = "Goutham Balaraman"
copyright = "2026, Goutham Balaraman"
version = release = sysone.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "numpydoc",
    "myst_nb",
    "sphinx_copybutton",
]

exclude_patterns = ["_build", "examples/.ipynb_checkpoints"]
# Single backticks in docstrings are code, as they read in the source.
default_role = "code"

# -- API reference ---------------------------------------------------------------------------------
autosummary_generate = True
autodoc_default_options = {"members": True, "inherited-members": True, "member-order": "bysource"}
autodoc_typehints = "none"  # the numpydoc sections describe the types
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_xref_param_type = False
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}

# -- Markdown and notebooks ------------------------------------------------------------------------
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3
nb_execution_mode = "off"  # the notebooks download models and need a GPU; they render without outputs

# The example notebooks live in examples/ at the repository root; copy them in for the build.
_examples = Path(__file__).parent / "examples"
_examples.mkdir(exist_ok=True)
for notebook in sorted((ROOT / "examples").glob("*.ipynb")):
    shutil.copy2(notebook, _examples / notebook.name)

# -- HTML ------------------------------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "sysone %s" % release
html_theme_options = {
    "github_url": "https://github.com/ForwardMetric/sysone",
    "icon_links": [{"name": "PyPI", "url": "https://pypi.org/project/sysone/", "icon": "fa-brands fa-python"}],
    "navigation_with_keys": False,
    "show_toc_level": 2,
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
}
html_context = {
    "github_user": "ForwardMetric",
    "github_repo": "sysone",
    "github_version": "main",
    "doc_path": "docs",
}
html_show_sourcelink = False
