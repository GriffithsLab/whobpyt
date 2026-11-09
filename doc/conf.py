"""Sphinx configuration for the WhoBPyT documentation.

Build locally with ``make html`` (or ``make html-noplot`` to skip running the
examples). See https://www.sphinx-doc.org/en/master/usage/configuration.html
for all available options.
"""

import os
from datetime import datetime, timezone

import matplotlib

matplotlib.use("agg")  # never open GUI windows while running examples

# -- Project information -----------------------------------------------------

project = "WhoBPyT"
copyright = f"2022–{datetime.now(tz=timezone.utc).year}, the WhoBPyT developers"  # noqa: A001

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "numpydoc",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]
root_doc = "index"
default_role = "py:obj"

# Strip shell / Python prompts when copying code blocks
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# Link to other projects' docs (e.g. ``torch.Tensor``, ``mne.Evoked``)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "mne": ("https://mne.tools/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- autodoc / numpydoc ------------------------------------------------------

# Show default argument values as written in the source (e.g. ``ones((200, 200))``)
# instead of dumping the full evaluated array into the rendered signature.
autodoc_preserve_defaults = True
numpydoc_class_members_toctree = False
numpydoc_show_inherited_class_members = False
numpydoc_attributes_as_param_list = True
numpydoc_xref_param_type = True
# Map the type names used in WhoBPyT docstrings to their documentation.
numpydoc_xref_aliases = {
    "Parameter": "whobpyt.datatypes.Parameter",
    "Timeseries": "whobpyt.datatypes.Timeseries",
    "Recording": "whobpyt.datatypes.Timeseries",
    "TrainingStats": "whobpyt.datatypes.TrainingStats",
    "AbstractParams": "whobpyt.datatypes.AbstractParams",
    "AbstractNeuralModel": "whobpyt.datatypes.AbstractNeuralModel",
    "AbstractNMM": "whobpyt.datatypes.AbstractNeuralModel",
    "AbstractLoss": "whobpyt.datatypes.AbstractLoss",
    "AbstractFitting": "whobpyt.datatypes.AbstractFitting",
    "JansenRitParams": "whobpyt.models.jansen_rit.JansenRitParams",
    "ParamsJR": "whobpyt.models.jansen_rit.JansenRitParams",
    "Tensor": "torch.Tensor",
    "torch.tensor": "torch.Tensor",
    "ndarray": "numpy.ndarray",
    "Int": "int",
    "Float": "float",
    "Bool": "bool",
    "String": "str",
}
# Words in type descriptions that are not types
numpydoc_xref_ignore = {
    "of", "or", "and", "default", "optional", "shape", "other", "x",
    "node_size", "output_size", "num_regions", "ts_length", "steps_per_TR",
    "TRs_per_window", "delays_max", "num_windows", "window_length",
}

# -- Sphinx-gallery ----------------------------------------------------------

sphinx_gallery_conf = {
    "examples_dirs": ["../examples"],
    "gallery_dirs": ["auto_examples"],
    "doc_module": ("whobpyt",),
    "backreferences_dir": "generated",
    "default_thumb_file": os.path.join("_static", "whobpyt_logo_feet.png"),
    "thumbnail_size": (160, 112),
    "within_subsection_order": "FileNameSortKey",
    "filename_pattern": r"^((?!sgskip).)*$",
    "remove_config_comments": True,
    "min_reported_time": 1.0,
    "line_numbers": False,
    "capture_repr": ("_repr_html_",),
    "matplotlib_animations": True,
    "reset_modules_order": "both",
    "show_memory": False,
    "show_api_usage": False,
    "copyfile_regex": r".*index\.rst",
    "abort_on_example_error": False,
    "plot_gallery": "True",
    "parallel": True,
}

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = project
html_baseurl = "https://griffithslab.github.io/whobpyt/"  # used for <link rel="canonical">
html_favicon = "_static/favicon.ico"
html_static_path = ["_static"]
html_css_files = ["style.css", "css/fix-sidebar.css"]
html_show_sourcelink = False
html_copy_source = False
html_show_sphinx = False

html_theme_options = {
    "logo": {
        "image_light": "_static/whobpyt_logo_feet.png",
        "image_dark": "_static/whobpyt_logo_feet_dark.png",
    },
    "icon_links": [
        dict(
            name="GitHub",
            url="https://github.com/griffithslab/whobpyt",
            icon="fa-brands fa-square-github fa-fw",
        ),
    ],
    "icon_links_label": "External Links",
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "navbar_align": "left",
    "navigation_with_keys": False,
    "navigation_depth": 4,
    "show_nav_level": 0,
    "show_toc_level": 1,
    "collapse_navigation": False,
    "article_header_start": [],  # no breadcrumbs
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "use_edit_page_button": False,
    "back_to_top_button": False,
    "footer_start": ["copyright"],
    # To enable Google Analytics, add this project's own measurement ID here:
    # "analytics": dict(google_analytics_id="G-XXXXXXXXXX"),
}

# Hide the left sidebar on the example pages so the code has room
html_sidebars = {"auto_examples/*": []}

# Logo widths (rem) for the homepage institution logos
_logo_size = "5"

html_context = {
    "default_mode": "auto",
    "github_user": "griffithslab",
    "github_repo": "whobpyt",
    "github_version": "main",
    "doc_path": "doc",
    # Rendered by _templates/homepage.html on the front page
    "carousel": [
        dict(
            title="Brain Dynamics",
            text="Oscillations, functional connectivity, and brain network organization.",
            url="documentation/science/brain_dynamics/index.html",
            img="homepage/brain_dynamics.jpg",
            alt="Whole-brain model of network recurrence across the cortical hierarchy (Momi et al. 2025)",
        ),
        dict(
            title="Brain Stimulation",
            text="Electrical, magnetic and acoustic, invasive and noninvasive brain stimulation.",
            url="documentation/science/brain_stimulation/index.html",
            img="homepage/brain_stimulation.jpg",
            alt="TMS-evoked responses spreading through a brain network (Momi et al. 2023)",
        ),
        dict(
            title="Brain & Cognition",
            text="Evoked and induced activity during cognitive tasks.",
            url="documentation/science/brain_cognition/index.html",
            img="homepage/brain_cognition.jpg",
            alt="Personalized brain network model of language lateralization (Ismail et al. 2026)",
        ),
    ],
    "institutions": [
        dict(name="University of Toronto", img="UofT_logo.jpg",
             url="https://www.utoronto.ca/", size=_logo_size),
        dict(name="Krembil Centre for Neuroinformatics", img="KCNI_logo.png",
             url="https://www.krembilneuroinformatics.ca/", size=_logo_size),
        dict(name="Centre for Addiction and Mental Health", img="CAMH_logo_purple.png",
             url="https://www.camh.ca/", size=_logo_size),
        dict(name="Hospital for Sick Children", img="sick_kids_logo.png",
             url="https://www.sickkids.ca/", size=_logo_size),
    ],
}
