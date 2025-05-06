"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import faulthandler
import os
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import metadata
from pathlib import Path

import matplotlib
import sphinx
#from intersphinx_registry import get_intersphinx_mapping
from numpydoc import docscrape
from sphinx.config import is_serializable
from sphinx.domains.changeset import versionlabels
from sphinx_gallery.sorting import ExplicitOrder

import whobpyt

#import mne
#import mne.html_templates._templates
#from mne.tests.test_docstring_parameters import error_ignores

#from mne.utils import (
#    linkcode_resolve,
#    run_subprocess,
#)

linkcode_resolve = None

# assert linkcode_resolve is not None  # avoid flake warnings, used by numpydoc
matplotlib.use("agg")
faulthandler.enable()
os.environ["_MNE_BROWSER_NO_BLOCK"] = "true"
os.environ["MNE_BROWSER_OVERVIEW_MODE"] = "hidden"
os.environ["MNE_BROWSER_THEME"] = "light"
os.environ["MNE_3D_OPTION_THEME"] = "light"
# https://numba.readthedocs.io/en/latest/reference/deprecation.html#deprecation-of-old-style-numba-captured-errors  # noqa: E501
os.environ["NUMBA_CAPTURED_ERRORS"] = "new_style"
#mne.html_templates._templates._COLLAPSED = True  # collapse info _repr_html_

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
curpath = Path(__file__).parent.resolve(strict=True)
sys.path.append(str(curpath / "sphinxext"))

# from credit_tools import generate_credit_rst  # noqa: E402
#from mne_doc_utils import report_scraper, reset_warnings, sphinx_logger  # noqa: E402

# -- Project information -----------------------------------------------------

project = "MNE"
td = datetime.now(tz=timezone.utc)

# We need to triage which date type we use so that incremental builds work
# (Sphinx looks at variable changes and rewrites all files if some change)
#copyright = (  # noqa: A001
#    f'2012–{td.year}, MNE Developers. Last updated <time datetime="{td.isoformat()}" class="localized">{td.strftime("%Y-%m-%d %H:%M %Z")}</time>\n'  # noqa: E501
#    '<script type="text/javascript">$(function () { $("time.localized").each(function () { var el = $(this); el.text(new Date(el.attr("datetime")).toLocaleString([], {dateStyle: "medium", timeStyle: "long"})); }); } )</script>'  # noqa: E501
#)

#if os.getenv("MNE_FULL_DATE", "false").lower() != "true":
#    copyright = f"2012–{td.year}, MNE Developers. Last updated locally."  # noqa: A001

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The full version, including alpha/beta/rc tags.
#release = whobpyt.__version__
#sphinx_logger.info(f"Building documentation for MNE {release} ({mne.__file__})")
# The short X.Y version.
#version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = "6.0"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # builtin
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.graphviz",
    # "sphinx.ext.intersphinx",
    #"sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    # contrib
    "matplotlib.sphinxext.plot_directive",
    "numpydoc",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.youtube",
    "sphinxcontrib.towncrier.ext",
    # homegrown
#    "contrib_avatars",
#    "gen_commands",
#    "gen_names",
#    "gh_substitutions",
#    "mne_substitutions",
#    "newcontrib_substitutions",
#   "unit_role",
#    "related_software",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.

# NB: changes here should also be made to the linkcheck target in the Makefile
exclude_patterns = ["_includes", "changes/devel"]

# The suffix of source filenames.
source_suffix = ".rst"

# The main toctree document.
master_doc = "index"

# List of documents that shouldn't be included in the build.
unused_docs = []

# List of directories, relative to source directory, that shouldn't be searched
# for source files.
exclude_trees = ["_build"]

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = "py:obj"

# A list of ignored prefixes for module index sorting.
modindex_common_prefix = [] # ["mne."]

# -- Sphinx-Copybutton configuration -----------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- sphinxcontrib-towncrier configuration -----------------------------------

towncrier_draft_working_directory = str(curpath.parent)

# -- Intersphinx configuration -----------------------------------------------

"""
intersphinx_mapping = {
    # More niche so didn't upstream to intersphinx_registry
    "nitime": ("https://nipy.org/nitime/", None),
    "mne_bids": ("https://mne.tools/mne-bids/stable", None),
    "mne-connectivity": ("https://mne.tools/mne-connectivity/stable", None),
    "mne-gui-addons": ("https://mne.tools/mne-gui-addons", None),
    "picard": ("https://pierreablin.github.io/picard/", None),
    "eeglabio": ("https://eeglabio.readthedocs.io/en/latest", None),
    "pybv": ("https://pybv.readthedocs.io/en/latest", None),
}
"""

#intersphinx_mapping.update(
#    get_intersphinx_mapping(
#        packages=set(
#            """
#imageio matplotlib numpy pandas python scipy statsmodels sklearn numba joblib nibabel
#seaborn patsy pyvista dipy nilearn pyqtgraph
#""".strip().split()
#        ),
#    )
#)


# NumPyDoc configuration -----------------------------------------------------

# Define what extra methods numpydoc will document
#docscrape.ClassDoc.extra_public_methods = mne.utils._doc_special_members
numpydoc_class_members_toctree = False
numpydoc_show_inherited_class_members = {}
#    "mne.Forward": False,
#    "mne.Projection": False,
#    "mne.SourceSpaces": False,
#}
numpydoc_attributes_as_param_list = True
numpydoc_xref_param_type = True

"""
numpydoc_xref_aliases = {
    # Python
    "file-like": ":term:`file-like <python:file object>`",
    "iterator": ":term:`iterator <python:iterator>`",
    "path-like": ":term:`path-like`",
    "array-like": ":term:`array_like <numpy:array_like>`",
    "Path": ":class:`python:pathlib.Path`",
    "bool": ":ref:`bool <python:typebool>`",
    # Matplotlib
    "colormap": ":ref:`colormap <matplotlib:colormaps>`",
    "color": ":doc:`color <matplotlib:api/colors_api>`",
    "Axes": "matplotlib.axes.Axes",
    "Figure": "matplotlib.figure.Figure",
    "Axes3D": "mpl_toolkits.mplot3d.axes3d.Axes3D",
    "ColorbarBase": "matplotlib.colorbar.ColorbarBase",
    # sklearn
    "LeaveOneOut": "sklearn.model_selection.LeaveOneOut",
    "MetadataRequest": "sklearn.utils.metadata_routing.MetadataRequest",
    "estimator": "sklearn.base.BaseEstimator",
    # joblib
    "joblib.Parallel": "joblib.Parallel",
    # nibabel
    "Nifti1Image": "nibabel.nifti1.Nifti1Image",
    "Nifti2Image": "nibabel.nifti2.Nifti2Image",
    "SpatialImage": "nibabel.spatialimages.SpatialImage",
    # MNE
    "Label": "mne.Label",
    "Forward": "mne.Forward",
    "Evoked": "mne.Evoked",
    "Info": "mne.Info",
    "SourceSpaces": "mne.SourceSpaces",
    "Epochs": "mne.Epochs",
    "Layout": "mne.channels.Layout",
    "EvokedArray": "mne.EvokedArray",
    "BiHemiLabel": "mne.BiHemiLabel",
    "AverageTFR": "mne.time_frequency.AverageTFR",
    "AverageTFRArray": "mne.time_frequency.AverageTFRArray",
    "EpochsTFR": "mne.time_frequency.EpochsTFR",
    "EpochsTFRArray": "mne.time_frequency.EpochsTFRArray",
    "RawTFR": "mne.time_frequency.RawTFR",
    "RawTFRArray": "mne.time_frequency.RawTFRArray",
    "Raw": "mne.io.Raw",
    "ICA": "mne.preprocessing.ICA",
    "Covariance": "mne.Covariance",
    "Annotations": "mne.Annotations",
    "DigMontage": "mne.channels.DigMontage",
    "VectorSourceEstimate": "mne.VectorSourceEstimate",
    "VolSourceEstimate": "mne.VolSourceEstimate",
    "VolVectorSourceEstimate": "mne.VolVectorSourceEstimate",
    "MixedSourceEstimate": "mne.MixedSourceEstimate",
    "MixedVectorSourceEstimate": "mne.MixedVectorSourceEstimate",
    "SourceEstimate": "mne.SourceEstimate",
    "Projection": "mne.Projection",
    "ConductorModel": "mne.bem.ConductorModel",
    "Dipole": "mne.Dipole",
    "DipoleFixed": "mne.DipoleFixed",
    "InverseOperator": "mne.minimum_norm.InverseOperator",
    "CrossSpectralDensity": "mne.time_frequency.CrossSpectralDensity",
    "SourceMorph": "mne.SourceMorph",
    "Xdawn": "mne.preprocessing.Xdawn",
    "Report": "mne.Report",
    "TimeDelayingRidge": "mne.decoding.TimeDelayingRidge",
    "Vectorizer": "mne.decoding.Vectorizer",
    "UnsupervisedSpatialFilter": "mne.decoding.UnsupervisedSpatialFilter",
    "TemporalFilter": "mne.decoding.TemporalFilter",
    "SSD": "mne.decoding.SSD",
    "Scaler": "mne.decoding.Scaler",
    "SPoC": "mne.decoding.SPoC",
    "PSDEstimator": "mne.decoding.PSDEstimator",
    "LinearModel": "mne.decoding.LinearModel",
    "FilterEstimator": "mne.decoding.FilterEstimator",
    "EMS": "mne.decoding.EMS",
    "CSP": "mne.decoding.CSP",
    "Beamformer": "mne.beamformer.Beamformer",
    "Transform": "mne.transforms.Transform",
    "Coregistration": "mne.coreg.Coregistration",
    "Figure3D": "mne.viz.Figure3D",
    "EOGRegression": "mne.preprocessing.EOGRegression",
    "Spectrum": "mne.time_frequency.Spectrum",
    "EpochsSpectrum": "mne.time_frequency.EpochsSpectrum",
    "EpochsFIF": "mne.Epochs",
    "EpochsEEGLAB": "mne.Epochs",
    "EpochsKIT": "mne.Epochs",
    "RawANT": "mne.io.Raw",
    "RawBOXY": "mne.io.Raw",
    "RawBrainVision": "mne.io.Raw",
    "RawBTi": "mne.io.Raw",
    "RawCTF": "mne.io.Raw",
    "RawCurry": "mne.io.Raw",
    "RawEDF": "mne.io.Raw",
    "RawEEGLAB": "mne.io.Raw",
    "RawEGI": "mne.io.Raw",
    "RawEximia": "mne.io.Raw",
    "RawEyelink": "mne.io.Raw",
    "RawFIL": "mne.io.Raw",
    "RawGDF": "mne.io.Raw",
    "RawHitachi": "mne.io.Raw",
    "RawKIT": "mne.io.Raw",
    "RawNedf": "mne.io.Raw",
    "RawNeuralynx": "mne.io.Raw",
    "RawNihon": "mne.io.Raw",
    "RawNIRX": "mne.io.Raw",
    "RawPersyst": "mne.io.Raw",
    "RawSNIRF": "mne.io.Raw",
    "Calibration": "mne.preprocessing.eyetracking.Calibration",
    # dipy
    "dipy.align.AffineMap": "dipy.align.imaffine.AffineMap",
    "dipy.align.DiffeomorphicMap": "dipy.align.imwarp.DiffeomorphicMap",
}
numpydoc_xref_ignore = {
    # words
    "and",
    "between",
    "instance",
    "instances",
    "of",
    "default",
    "shape",
    "or",
    "with",
    "length",
    "pair",
    "matplotlib",
    "optional",
    "kwargs",
    "in",
    "dtype",
    "object",
    # shapes
    "n_vertices",
    "n_faces",
    "n_channels",
    "m",
    "n",
    "n_events",
    "n_colors",
    "n_times",
    "obj",
    "n_chan",
    "n_epochs",
    "n_picks",
    "n_ch_groups",
    "n_dipoles",
    "n_ica_components",
    "n_pos",
    "n_node_names",
    "n_tapers",
    "n_signals",
    "n_step",
    "n_freqs",
    "wsize",
    "Tx",
    "M",
    "N",
    "p",
    "q",
    "r",
    "n_observations",
    "n_regressors",
    "n_cols",
    "n_frequencies",
    "n_tests",
    "n_samples",
    "n_peaks",
    "n_permutations",
    "nchan",
    "n_points",
    "n_features",
    "n_parts",
    "n_features_new",
    "n_components",
    "n_labels",
    "n_events_in",
    "n_splits",
    "n_scores",
    "n_outputs",
    "n_trials",
    "n_estimators",
    "n_tasks",
    "nd_features",
    "n_classes",
    "n_targets",
    "n_slices",
    "n_hpi",
    "n_fids",
    "n_elp",
    "n_pts",
    "n_tris",
    "n_nodes",
    "n_nonzero",
    "n_events_out",
    "n_segments",
    "n_orient_inv",
    "n_orient_fwd",
    "n_orient",
    "n_dipoles_lcmv",
    "n_dipoles_fwd",
    "n_picks_ref",
    "n_coords",
    "n_meg",
    "n_good_meg",
    "n_moments",
    "n_patterns",
    "n_new_events",
    # sklearn subclasses
    "mapping",
    "to",
    "any",
    "pandas",
    "polars",
    "default",
    # unlinkable
    "CoregistrationUI",
    "mne_qt_browser.figure.MNEQtBrowser",
    # pooch, since its website is unreliable and users will rarely need the links
    "pooch.Unzip",
    "pooch.Untar",
    "pooch.HTTPDownloader",
}
numpydoc_validate = True
numpydoc_validation_checks = {"all"} | set(error_ignores)
numpydoc_validation_exclude = {  # set of regex
    # dict subclasses
    r"\.clear",
    r"\.get$",
    r"\.copy$",
    r"\.fromkeys",
    r"\.items",
    r"\.keys",
    r"\.move_to_end",
    r"\.pop",
    r"\.popitem",
    r"\.setdefault",
    r"\.update",
    r"\.values",
    # list subclasses
    r"\.append",
    r"\.count",
    r"\.extend",
    r"\.index",
    r"\.insert",
    r"\.remove",
    r"\.sort",
    # we currently don't document these properly (probably okay)
    r"\.__getitem__",
    r"\.__contains__",
    r"\.__hash__",
    r"\.__mul__",
    r"\.__sub__",
    r"\.__add__",
    r"\.__iter__",
    r"\.__div__",
    r"\.__neg__",
    # copied from sklearn
    r"mne\.utils\.deprecated",
}
"""

# -- Sphinx-gallery configuration --------------------------------------------

examples_dirs = ["../examples"] # ["../tutorials", "../examples"]
gallery_dirs = ["auto_examples"] # ["auto_tutorials", "auto_examples"]
#os.environ["_MNE_BUILDING_DOC"] = "true"

#scrapers = (
#    "matplotlib",
#    "mne_doc_utils.gui_scraper",
#    "mne_doc_utils.brain_scraper",
#    "pyvista",
#    "mne_doc_utils.report_scraper",
#    "mne_doc_utils.mne_qt_browser_scraper",
#)

compress_images = ("images", "thumbnails")
# let's make things easier on Windows users
# (on Linux and macOS it's easy enough to require this)
if sys.platform.startswith("win"):
    try:
        subprocess.check_call(["optipng", "--version"])
    except Exception:
        compress_images = ()

#sphinx_gallery_parallel = int(os.getenv("MNE_DOC_BUILD_N_JOBS", "1"))
sphinx_gallery_conf = {
    "doc_module": ("whobpyt",),
    "reference_url": dict(whobpyt=None),
    "examples_dirs": examples_dirs,
    "subsection_order": ExplicitOrder(
        [
            "../examples/eg__tmseeg",
            "../examples/eg__Ismail2025",
            "../examples/eg__momi2023",
            "../examples/eg__momi2025",
            "../examples/visualization",
            "../examples/preprocessing"
        ]
    ),
    "gallery_dirs": gallery_dirs,
    "default_thumb_file": os.path.join("_static", "whobpyt_logo_feet.png"),
    "backreferences_dir": "generated",
    "plot_gallery": "True",  # Avoid annoying Unicode/bool default warning
    "thumbnail_size": (160, 112),
    "remove_config_comments": True,
    "min_reported_time": 1.0,
    "abort_on_example_error": False,
    #"reset_modules": (
    #    "matplotlib",
    #    "mne_doc_utils.reset_modules",
    #),  # called w/each script
    "reset_modules_order": "both",
    #"image_scrapers": scrapers,
    "show_memory": sys.platform == "linux" , #and sphinx_gallery_parallel == 1,
    "line_numbers": False,  # messes with style
    "within_subsection_order": "FileNameSortKey",
    "capture_repr": ("_repr_html_",),
    "junit": os.path.join("..", "test-results", "sphinx-gallery", "junit.xml"),
    "matplotlib_animations": True,
    "compress_images": compress_images,
    "filename_pattern": "^((?!sgskip).)*$",
    "exclude_implicit_doc": {
     #   r"mne\.io\.read_raw_fif",
     #   r"mne\.io\.Raw",
     #   r"mne\.Epochs",
     #   r"mne.datasets.*",
    },
    "show_api_usage": "unused",
    "copyfile_regex": r".*index\.rst" #,  # allow custom index.rst files
    }
    #"api_usage_ignore": (),
    #
    #    "("
    #    ".*__.*__|"  # built-ins
    #    ".*Base.*|.*Array.*|mne.Vector.*|mne.Mixed.*|mne.Vol.*|"  # inherited
    #    "mne.coreg.Coregistration.*|"  # GUI
    #    # common
    #    ".*utils.*|.*verbose()|.*copy()|.*update()|.*save()|"
    #    ".*get_data()|"
    #    # mixins
    #    ".*add_channels()|.*add_reference_channels()|"
    #    ".*anonymize()|.*apply_baseline()|.*apply_function()|"
    #    ".*apply_hilbert()|.*as_type()|.*decimate()|"
    #    ".*drop()|.*drop_channels()|.*drop_log_stats()|"
    #    ".*export()|.*get_channel_types()|"
    #    ".*get_montage()|.*interpolate_bads()|.*next()|"
    #    ".*pick()|.*pick_channels()|.*pick_types()|"
    #    ".*plot_sensors()|.*rename_channels()|"
    #    ".*reorder_channels()|.*savgol_filter()|"
    #    ".*set_eeg_reference()|.*set_channel_types()|"
    #    ".*set_meas_date()|.*set_montage()|.*shift_time()|"
    #    ".*time_as_index()|.*to_data_frame()|"
    #    # dictionary inherited
    #    ".*clear()|.*fromkeys()|.*get()|.*items()|"
    #    ".*keys()|.*pop()|.*popitem()|.*setdefault()|"
    #    ".*values()|"
    #    # sklearn inherited
    #    ".*apply()|.*decision_function()|.*fit()|"
    #    ".*fit_transform()|.*get_params()|.*predict()|"
    #    ".*predict_proba()|.*set_params()|.*transform()|"
    #    # I/O, also related to mixins
    #    ".*.remove.*|.*.write.*)"
    #),
    #"parallel": sphinx_gallery_parallel,

assert is_serializable(sphinx_gallery_conf)
# Files were renamed from plot_* with:
# find . -type f -name 'plot_*.py' -exec sh -c 'x="{}"; xn=`basename "${x}"`; git mv "$x" `dirname "${x}"`/${xn:5}' \;  # noqa


def append_attr_meth_examples(app, what, name, obj, options, lines):
    """Append SG examples backreferences to method and attr docstrings."""
    # NumpyDoc nicely embeds method and attribute docstrings for us, but it
    # does not respect the autodoc templates that would otherwise insert
    # the .. include:: lines, so we need to do it.
    # Eventually this could perhaps live in SG.
    if what in ("attribute", "method"):
        size = os.path.getsize(
            os.path.join(
                os.path.dirname(__file__),
                "generated",
                f"{name}.examples",
            )
        )
        if size > 0:
            lines += """
.. _sphx_glr_backreferences_{1}:

.. rubric:: Examples using ``{0}``:

.. minigallery:: {1}

""".format(name.split(".")[-1], name).split("\n")


#def fix_sklearn_inherited_docstrings(app, what, name, obj, options, lines):
#    """Fix sklearn docstrings because they use autolink and we do not."""
#    if (
#        name.startswith("mne.decoding.") or name.startswith("mne.preprocessing.Xdawn")
#    ) and name.endswith(
#        (
#            ".get_metadata_routing",
#            ".fit",
#            ".fit_transform",
#            ".set_output",
#            ".transform",
#        )
#    ):
#        if ":Parameters:" in lines:
#            loc = lines.index(":Parameters:")
#        else:
#            loc = lines.index(":Returns:")
#        lines.insert(loc, "")
#        lines.insert(loc, ".. default-role:: autolink")
#        lines.insert(loc, "")


# -- Other extension configuration -------------------------------------------

# Consider using http://magjac.com/graphviz-visual-editor for this
graphviz_dot_args = [
    "-Gsep=-0.5",
    "-Gpad=0.5",
    "-Nshape=box",
    "-Nfontsize=20",
    "-Nfontname=Open Sans,Arial",
]
graphviz_output_format = "svg"  # for API usage diagrams
user_agent = "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Mobile Safari/537.36"  # noqa: E501
# Can eventually add linkcheck_request_headers if needed
linkcheck_ignore = [  # will be compiled to regex
    # 403 Client Error: Forbidden
    "https://doi.org/10.1002/",  # onlinelibrary.wiley.com/doi/10.1002/hbm
    "https://doi.org/10.1016/",  # neuroimage
    "https://doi.org/10.1021/",  # pubs.acs.org/doi/abs
    "https://doi.org/10.1063/",  # pubs.aip.org/aip/jap
    "https://doi.org/10.1073/",  # pnas.org
    "https://doi.org/10.1080/",  # www.tandfonline.com
    "https://doi.org/10.1088/",  # www.tandfonline.com
    "https://doi.org/10.1093/",  # academic.oup.com/sleep/
    "https://doi.org/10.1098/",  # royalsocietypublishing.org
    "https://doi.org/10.1101/",  # www.biorxiv.org
    "https://doi.org/10.1103",  # journals.aps.org/rmp
    "https://doi.org/10.1111/",  # onlinelibrary.wiley.com/doi/10.1111/psyp
    "https://doi.org/10.1126/",  # www.science.org
    "https://doi.org/10.1137/",  # epubs.siam.org
    "https://doi.org/10.1145/",  # dl.acm.org
    "https://doi.org/10.1155/",  # www.hindawi.com/journals/cin
    "https://doi.org/10.1161/",  # www.ahajournals.org
    "https://doi.org/10.1162/",  # direct.mit.edu/neco/article/
    "https://doi.org/10.1167/",  # jov.arvojournals.org
    "https://doi.org/10.1177/",  # journals.sagepub.com
    "https://doi.org/10.3109/",  # www.tandfonline.com
    "https://www.biorxiv.org/content/10.1101/",  # biorxiv.org
    "https://www.researchgate.net/profile/",
    "https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html",
    r"https://scholar.google.com/scholar\?cites=12188330066413208874&as_ylo=2014",
    r"https://scholar.google.com/scholar\?cites=1521584321377182930&as_ylo=2013",
    "https://www.research.chop.edu/imaging",
    "http://prdownloads.sourceforge.net/optipng",
    "https://sourceforge.net/projects/aespa/files/",
    "https://sourceforge.net/projects/ezwinports/files/",
    "https://www.mathworks.com/products/compiler/matlab-runtime.html",
    "https://medicine.umich.edu/dept/khri/ross-maddox-phd",
    # 500 server error
    "https://openwetware.org/wiki/Beauchamp:FreeSurfer",
    # 503 Server error
    "https://hal.archives-ouvertes.fr/hal-01848442",
    # Read timed out
    "http://www.cs.ucl.ac.uk/staff/d.barber/brml",
    "https://www.cea.fr",
    "http://www.humanconnectome.org/data",
    "https://www.mail-archive.com/freesurfer@nmr.mgh.harvard.edu",
    "https://launchpad.net",
    # Max retries exceeded
    "https://doi.org/10.7488/ds/1556",
    "https://datashare.is.ed.ac.uk/handle/10283",
    "https://imaging.mrc-cbu.cam.ac.uk/imaging/MniTalairach",
    "https://www.nyu.edu/",
    # Too slow
    "https://speakerdeck.com/dengemann/",
    "https://www.dtu.dk/english/service/phonebook/person",
    "https://www.gnu.org/software/make/",
    "https://www.macports.org/",
    "https://hastie.su.domains/CASI",
    # SSL problems sometimes
    "http://ilabs.washington.edu",
    "https://psychophysiology.cpmc.columbia.edu",
    "https://erc.easme-web.eu",
    # Not rendered by linkcheck builder
    r"ides\.html",
]
linkcheck_anchors = False  # saves a bit of time
linkcheck_timeout = 15  # some can be quite slow
linkcheck_retries = 3
linkcheck_report_timeouts_as_broken = False

# autodoc / autosummary
autosummary_generate = True
autodoc_default_options = {"inherited-members": None}

# sphinxcontrib-bibtex
bibtex_bibfiles = ["./references.bib"]
bibtex_style = "unsrt"
bibtex_footbibliography_header = ""


# -- Nitpicky ----------------------------------------------------------------

nitpicky = True
show_warning_types = True
nitpick_ignore = [
    ("py:class", "None.  Remove all items from D."),
    ("py:class", "a set-like object providing a view on D's items"),
    ("py:class", "a set-like object providing a view on D's keys"),
    (
        "py:class",
        "v, remove specified key and return the corresponding value.",
    ),  # noqa: E501
    ("py:class", "None.  Update D from dict/iterable E and F."),
    ("py:class", "an object providing a view on D's values"),
    ("py:class", "a shallow copy of D"),
    ("py:class", "(k, v), remove and return some (key, value) pair as a"),
    ("py:class", "_FuncT"),  # type hint used in @verbose decorator
#    ("py:class", "mne.utils._logging._FuncT"),
    ("py:class", "None.  Remove all items from od."),
]
nitpick_ignore_regex = [
    # Classes whose methods we purposefully do not document
#    ("py:.*", r"mne\.io\.BaseRaw.*"),  # use mne.io.Raw
#    ("py:.*", r"mne\.BaseEpochs.*"),  # use mne.Epochs
    # Type hints for undocumented types
#    ("py:.*", r"mne\.io\..*\.Raw.*"),  # RawEDF etc.
#    ("py:.*", r"mne\.epochs\.EpochsFIF.*"),
#    ("py:.*", r"mne\.io\..*\.Epochs.*"),  # EpochsKIT etc.
    (  # BaseRaw attributes are documented in Raw
        "py:obj",
        "(filename|metadata|proj|times|tmax|tmin|annotations|ch_names"
        "|compensation_grade|duration|filenames|first_samp|first_time"
        "|last_samp|n_times|proj|times|tmax|tmin)",
    ),
]
suppress_warnings = [
    "image.nonlocal_uri",  # we intentionally link outside
]


# -- Sphinx hacks / overrides ------------------------------------------------

versionlabels["versionadded"] = sphinx.locale._("New in v%s")

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
# switcher_version_match = "dev" if ".dev" in version else version
html_theme_options = {
    "icon_links": [
        dict(
            name="GitHub",
            url="https://github.com/griffithslab/whobpyt",
            icon="fa-brands fa-square-github fa-fw",
        ),
    ],
    "icon_links_label": "External Links",  # for screen reader
    "use_edit_page_button": False,
    "navigation_with_keys": False,
    "show_toc_level": 1,
    "article_header_start": [],  # disable breadcrumbs
    "navbar_end": [
        "theme-switcher", #"version-switcher",
        "navbar-icon-links",
    ],
    "navbar_align": "left",
    "navbar_persistent": ["search-button"],
    #"footer_start": ["copyright"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "analytics": dict(google_analytics_id="G-5TBCPCRB6X"),
    #"switcher": { "json_url": "https://mne.tools/dev/_static/versions.json", "version_match": switcher_version_match,
    "back_to_top_button": False,
    "logo": {
        "image_light": "_static/whobpyt_logo_feet.png",
        "image_dark": "_static/whobpyt_logo_feet_dark.png",
    }
}

## The name of an image file (relative to this directory) to place at the top
## of the sidebar.
#html_logo = "_static/whobpyt_logo_feet.png"
#html_theme_options = {
#    "logo": {
#        "image_light": "_static/whobpyt_logo_feet.png",
#        "image_dark": "_static/whobpyt_logo_feet_dark.png",
#    }
#}



# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "_static/favicon.ico"


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = [
    "style.css",
]

# # Add any extra paths that contain custom files (such as robots.txt or
# # .htaccess) here, relative to this directory. These files are copied
# # directly to the root of the documentation.
# html_extra_path = [
#     "contributing.html",
#     "documentation.html",
#     "getting_started.html",
#     "install_mne_python.html",
# ]

# Custom sidebar templates, maps document names to template names.
html_sidebars = {} #{
#    "index": ["sidebar-quicklinks.html"],
#}

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False
html_copy_source = False

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

# accommodate different logo shapes (width values in rem)
xs = "2"
sm = "2.5"
md = "3"
lg = "4.5"
xl = "5"
xxl = "6"
# variables to pass to HTML templating engine
html_context = {
    "default_mode": "auto",
    # next 3 are for the "edit this page" button
    "github_user": "griffithslab",
    "github_repo": "whobpyt",
    "github_version": "main",
    "doc_path": "doc",
    #"funders": [ dict(img="nih.svg", size="3", title="CAMH"), dict(img="nsf.png", size="3.5", title="CIHR"),dict( img="erc.svg", size="3.5", title="UofT", klass="only-light"),],
    "institutions": [
        dict(
            name="University of Toronto",
            img="UofT_logo.jpg",
            url="https://www.utoronto.ca/",
            size=xl,
        ),
        dict(
            name="Krembil Centre for Neuroinformatics",
            img="KCNI_logo.png",
            url="https://www.krembilneuroinformatics.ca/",
            size=xl,
        ),
        dict(
            name="Center for Addiction and Mental Health",
            img="CAMH_logo_purple.png",
            url="https://www.camh.ca/",
            size=xl,
        ),
        dict(
            name="Hospital for Sick Children",
            img="sick_kids_logo.png",
            url="https://www.sickkids.ca/",
            size=xl,
        )
    ],
    # \u00AD is an optional hyphen (not rendered unless needed)
    # If these are changed, the Makefile should be updated, too
    "carousel": [
        dict(
            title="Brain Dynamics",
            text="oscillations, functional connectivity, brain network organization", 
            url="auto_examples/index.html",
            img="../_static/Ismail2025_SummaryFigure.png",
            alt="dSPM",
        ),
        dict(
            title="Brain Stimulation",
            text="electrical/magnetic/acoustic, invasive/noninvasive brain stimulation",
            url="auto_examples/index.html",
            img="../_static/momi2025__SummaryFigure.png",
            alt="Stimulation",
        ),
        dict(
            title="Brain & Cognition",
            text="evoked/induced activity during cognitive tasks",
            url="auto_examples/index.html",
            img="../_static/Ismail2025_SummaryFigure.png",
            alt="Cognition",
        ),
    ],
}

html_theme = "pydata_sphinx_theme"



# Output file base name for HTML help builder.
htmlhelp_basename = "whobpyt-doc"


# -- Options for plot_directive ----------------------------------------------

# Adapted from SciPy
plot_include_source = True
plot_formats = [("png", 96)]
plot_html_show_formats = False
plot_html_show_source_link = False
font_size = 13 * 72 / 96.0  # 13 px
plot_rcparams = {
    "font.size": font_size,
    "axes.titlesize": font_size,
    "axes.labelsize": font_size,
    "xtick.labelsize": font_size,
    "ytick.labelsize": font_size,
    "legend.fontsize": font_size,
    "figure.figsize": (6, 5),
    "figure.subplot.bottom": 0.2,
    "figure.subplot.left": 0.2,
    "figure.subplot.right": 0.9,
    "figure.subplot.top": 0.85,
    "figure.subplot.wspace": 0.4,
    "text.usetex": False,
}


# -- Options for LaTeX output ------------------------------------------------

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass
# [howto/manual]).
latex_documents = []

# The name of an image file (relative to this directory) to place at the top of
# the title page.
latex_logo = "_static/logo.png"

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
latex_toplevel_sectioning = "part"

# -- Warnings management -----------------------------------------------------
#reset_warnings(None, None)

# -- Fontawesome support -----------------------------------------------------
brand_icons = ("apple", "linux", "windows", "discourse", "python")
fixed_width_icons = (
    # homepage:
    "book",
    #"code-branch",
    "newspaper",
    "circle-question",
    "quote-left",
    # contrib guide:
    "bug-slash",
    "comment",
    "computer-mouse",
    "hand-sparkles",
    "pencil",
    "text-slash",
    "universal-access",
    "wand-magic-sparkles",
    "discourse",
    "python",
)
other_icons = (
    "hand-paper",
    "question",
    "rocket",
    "server",
    "code",
    "desktop",
    "terminal",
    "cloud-arrow-down",
    "wrench",
    "hourglass-half",
)
icon_class = dict()
for icon in brand_icons + fixed_width_icons + other_icons:
    icon_class[icon] = ("fa-brands",) if icon in brand_icons else ("fa-solid",)
    icon_class[icon] += ("fa-fw",) if icon in fixed_width_icons else ()

rst_prolog = ""
for icon, classes in icon_class.items():
    rst_prolog += f"""
.. |{icon}| raw:: html

    <i class="{" ".join(classes + (f"fa-{icon}",))}"></i>
"""

#rst_prolog += """
#.. |ensp| unicode:: U+2002 .. EN SPACE
#
#.. include:: /links.inc
#.. include:: /changes/names.inc
#
#.. currentmodule:: mne
#"""

# -- Dependency info ----------------------------------------------------------

min_py = metadata("whobpyt")["Requires-Python"].lstrip(" =<>")
rst_prolog += f"\n.. |min_python_version| replace:: {min_py}\n"

# -- website redirects --------------------------------------------------------

# Static list created 2021/04/13 based on what we needed to redirect,
# since we don't need to add redirects for examples added after this date.
needed_plot_redirects = {
#    # tutorials
#    "10_epochs_overview.py",
#    "10_evoked_overview.py",
#    "10_overview.py",
#    "10_preprocessing_overview.py",
#    "10_raw_overview.py",
#    "10_reading_meg_data.py",
#    "15_handling_bad_channels.py",
#    "20_event_arrays.py",
#    "20_events_from_raw.py",
#    "20_reading_eeg_data.py",
#    "20_rejecting_bad_data.py",
#    "20_visualize_epochs.py",
#    "20_visualize_evoked.py",
#    "30_annotate_raw.py",
#    "30_epochs_metadata.py",
#    "30_filtering_resampling.py",
#    "30_info.py",
#    "30_reading_fnirs_data.py",
#    "35_artifact_correction_regression.py",
#    "40_artifact_correction_ica.py",
#    "40_autogenerate_metadata.py",
#    "40_sensor_locations.py",
#    "40_visualize_raw.py",
#    "45_projectors_background.py",
#    "50_artifact_correction_ssp.py",
#    "50_configure_mne.py",
#    "50_epochs_to_data_frame.py",
#    "55_setting_eeg_reference.py",
#    "59_head_positions.py",
#    "60_make_fixed_length_epochs.py",
#    "60_maxwell_filtering_sss.py",
#    "70_fnirs_processing.py",
#    # examples
#    "3d_to_2d.py",
#    "brainstorm_data.py",
#    "channel_epochs_image.py",
#    "cluster_stats_evoked.py",
#    "compute_csd.py",
#    "compute_mne_inverse_epochs_in_label.py",
#    "compute_mne_inverse_raw_in_label.py",
#    "compute_mne_inverse_volume.py",
#    "compute_source_psd_epochs.py",
#    "covariance_whitening_dspm.py",
#    "custom_inverse_solver.py",
#    "decoding_csp_eeg.py",
#    "decoding_csp_timefreq.py",
#    "decoding_spatio_temporal_source.py",
#    "decoding_spoc_CMC.py",
#    "decoding_time_generalization_conditions.py",
#    "decoding_unsupervised_spatial_filter.py",
#    "decoding_xdawn_eeg.py",
#    "define_target_events.py",
#    "dics_source_power.py",
#    "eeg_csd.py",
#    "eeg_on_scalp.py",
#    "eeglab_head_sphere.py",
#    "elekta_epochs.py",
#    "ems_filtering.py",
#    "eog_artifact_histogram.py",
#    "evoked_arrowmap.py",
#    "evoked_ers_source_power.py",
#    "evoked_topomap.py",
#    "evoked_whitening.py",
#    "fdr_stats_evoked.py",
#    "find_ref_artifacts.py",
#    "fnirs_artifact_removal.py",
#    "forward_sensitivity_maps.py",
#    "gamma_map_inverse.py",
#    "hf_sef_data.py",
#    "ica_comparison.py",
#    "interpolate_bad_channels.py",
#    "label_activation_from_stc.py",
#    "label_from_stc.py",
#    "label_source_activations.py",
#    "left_cerebellum_volume_source.py",
#    "limo_data.py",
#    "linear_model_patterns.py",
#    "linear_regression_raw.py",
#    "meg_sensors.py",
#    "mixed_norm_inverse.py",
#    "mixed_source_space_inverse.py",
#    "mne_cov_power.py",
#    "mne_helmet.py",
#    "mne_inverse_coherence_epochs.py",
#    "mne_inverse_envelope_correlation.py",
#    "mne_inverse_envelope_correlation_volume.py",
#    "mne_inverse_psi_visual.py",
#    "morph_surface_stc.py",
#    "morph_volume_stc.py",
#    "movement_compensation.py",
#    "movement_detection.py",
#    "multidict_reweighted_tfmxne.py",
#    "muscle_detection.py",
#    "opm_data.py",
#    "otp.py",
#    "parcellation.py",
#    "psf_ctf_label_leakage.py",
#    "psf_ctf_vertices.py",
#    "psf_ctf_vertices_lcmv.py",
#    "publication_figure.py",
#    "rap_music.py",
#    "trap_music.py",
#    "read_inverse.py",
#    "read_neo_format.py",
#    "read_noise_covariance_matrix.py",
#    "read_stc.py",
#    "receptive_field_mtrf.py",
#    "resolution_metrics.py",
#    "resolution_metrics_eegmeg.py",
#    "roi_erpimage_by_rt.py",
#    "sensor_noise_level.py",
#    "sensor_permutation_test.py",
#    "sensor_regression.py",
#    "shift_evoked.py",
#    "simulate_evoked_data.py",
#    "simulate_raw_data.py",
#    "simulated_raw_data_using_subject_anatomy.py",
#    "snr_estimate.py",
#    "source_label_time_frequency.py",
#    "source_power_spectrum.py",
#    "source_power_spectrum_opm.py",
#    "source_simulator.py",
#    "source_space_morphing.py",
#    "source_space_snr.py",
#    "source_space_time_frequency.py",
#    "ssd_spatial_filters.py",
#    "ssp_projs_sensitivity_map.py",
#    "temporal_whitening.py",
#    "time_frequency_erds.py",
#    "time_frequency_global_field_power.py",
#    "time_frequency_mixed_norm_inverse.py",
#    "time_frequency_simulated.py",
#    "topo_compare_conditions.py",
#    "topo_customized.py",
#    "vector_mne_solution.py",
#    "virtual_evoked.py",
#    "xdawn_denoising.py",
#    "xhemi.py",
}
api_redirects = {
#    "connectivity",
#    "covariance",
#    "creating_from_arrays",
#    "datasets",
#    "decoding",
#    "events",
#    "export",
#    "file_io",
#    "forward",
#    "inverse",
#    "logging",
#    "most_used_classes",
#    "mri",
#    "preprocessing",
#    "python_reference",
#    "reading_raw_data",
#    "realtime",
#    "report",
#    "sensor_space",
#    "simulation",
#    "source_space",
#    "statistics",
#    "time_frequency",
#    "visualization",
}
ex = "auto_examples"
co = "connectivity"
mne_conn = "https://mne.tools/mne-connectivity/stable"
tu = "auto_tutorials"
pr = "preprocessing"
di = "discussions"
sm = "source-modeling"
fw = "forward"
nv = "inverse"
sn = "stats-sensor-space"
sr = "stats-source-space"
sd = "sample-datasets"
ml = "machine-learning"
tf = "time-freq"
si = "simulation"
vi = "visualization"
custom_redirects = {}
    # Custom redirects (one HTML path to another, relative to outdir)
    # can be added here as fr->to key->value mappings
#    "install/contributing": "development/contributing",
#    "overview/cite": "documentation/cite",
#    "overview/get_help": "help/index",
#    "overview/roadmap": "development/roadmap",
#    "whats_new": "development/whats_new",
#    f"{tu}/evoked/plot_eeg_erp": f"{tu}/evoked/30_eeg_erp",
#    f"{tu}/evoked/plot_whitened": f"{tu}/evoked/40_whitened",
#    f"{tu}/misc/plot_modifying_data_inplace": f"{tu}/intro/15_inplace",


# Adapted from sphinxcontrib/redirects (BSD-2-Clause)
REDIRECT_TEMPLATE = """\
<!DOCTYPE HTML>
<html lang="en-US">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="refresh" content="1; url={to}">
        <script type="text/javascript">
            window.location.href = "{to}"
        </script>
        <title>Page Redirection</title>
    </head>
    <body>
        If you are not redirected automatically, follow this <a href='{to}'>link</a>.
    </body>
</html>"""


def check_existing_redirect(path):
    """Make sure existing HTML files are redirects, before overwriting."""
    if path.is_file():
        with open(path) as fid:
            for _ in range(8):
                next(fid)
            line = fid.readline()
            if "Page Redirection" not in line:
                raise RuntimeError(
                    "Attempted overwrite of HTML file with a redirect, where the "
                    "original file was not already a redirect."
                )


def _check_valid_builder(app, exception):
    valid_builder = isinstance(app.builder, sphinx.builders.html.StandaloneHTMLBuilder)
    return valid_builder and exception is None


def make_gallery_redirects(app, exception):
    """Make HTML redirects for our sphinx gallery pages."""
    if not _check_valid_builder(app, exception):
        return
    sg_conf = app.config["sphinx_gallery_conf"]
    for src_dir, out_dir in zip(sg_conf["examples_dirs"], sg_conf["gallery_dirs"]):
        root = (Path(app.srcdir) / src_dir).resolve()
        fnames = [
            pyfile.relative_to(root)
            for pyfile in root.rglob(r"**/*.py")
            if pyfile.name in needed_plot_redirects
        ]
        # plot_ redirects
        for fname in fnames:
            dirname = Path(app.outdir) / out_dir / fname.parent
            to_fname = fname.with_suffix(".html").name
            fr_fname = f"plot_{to_fname}"
            to_path = dirname / to_fname
            fr_path = dirname / fr_fname
            assert to_path.is_file(), (fname, to_path)
            with open(fr_path, "w") as fid:
                fid.write(REDIRECT_TEMPLATE.format(to=to_fname))
        
        #sphinx_logger.info(
        #    f"Added {len(fnames):3d} HTML plot_* redirects for {out_dir}"
        #)


def make_api_redirects(app, exception):
    """Make HTML redirects for our API pages."""
    if not _check_valid_builder(app, exception):
        return

    for page in api_redirects:
        fname = f"{page}.html"
        fr_path = Path(app.outdir) / fname
        to_path = Path(app.outdir) / "api" / fname
        # allow overwrite if existing file is just a redirect
        check_existing_redirect(fr_path)
        with open(fr_path, "w") as fid:
            fid.write(REDIRECT_TEMPLATE.format(to=to_path))
    #sphinx_logger.info(f"Added {len(api_redirects):3d} HTML API redirects")


#def make_custom_redirects(app, exception):
#    """Make HTML redirects for miscellaneous pages."""
##    if not _check_valid_builder(app, exception):
#        return

#    for _fr, _to in custom_redirects.items():
#        fr = f"{_fr}.html"
#        to = f"{_to}.html"
#        fr_path = Path(app.outdir) / fr
#        check_existing_redirect(fr_path)
#        if to.startswith("http"):
#            to_path = to
#        else:
#            to_path = Path(app.outdir) / to
#            assert to_path.is_file(), to_path
#        # recreate folders that no longer exist
#        defunct_gallery_folders = ()
       #     "misc",
       #     "discussions",
       #     "source-modeling",
       #     "sample-datasets",
       #     "connectivity",
       # )
#        parts = fr_path.relative_to(Path(app.outdir)).parts
#        if (
#            len(parts) > 1  # whats_new violates this
#            and parts[1] in defunct_gallery_folders
#            and not fr_path.parent.exists()
#        ):
#            os.makedirs(fr_path.parent, exist_ok=True)
#        # write the redirect
#        with open(fr_path, "w") as fid:
#            fid.write(REDIRECT_TEMPLATE.format(to=to_path))
#    #sphinx_logger.info(f"Added {len(custom_redirects):3d} HTML custom redirects")


#def make_version(app, exception):
#    """Make a text file with the git version."""
#    if not (
#        isinstance(app.builder, sphinx.builders.html.StandaloneHTMLBuilder)
#        and exception is None
#    ):
#        return
#    try:
#        stdout, _ = run_subprocess(["git", "rev-parse", "HEAD"], verbose=False)
#    except Exception as exc:
#        #sphinx_logger.warning(f"Failed to write _version.txt: {exc}")
#        return
#    with open(os.path.join(app.outdir, "_version.txt"), "w") as fid:
#        fid.write(stdout)
#    #sphinx_logger.info(f'Added "{stdout.rstrip()}" > _version.txt')


# -- Connect our handlers to the main Sphinx app ---------------------------


#def setup(app):
    #"""Set up the Sphinx app."""
    #app.connect("autodoc-process-docstring", append_attr_meth_examples)
    #app.connect("autodoc-process-docstring", fix_sklearn_inherited_docstrings)
    # High prio, will happen before SG
    #app.connect("builder-inited", generate_credit_rst, priority=10)
    #app.connect("builder-inited", report_scraper.set_dirs, priority=20)
    #app.connect("build-finished", make_gallery_redirects)
    #app.connect("build-finished", make_api_redirects)
    #app.connect("build-finished", make_custom_redirects)
    #app.connect("build-finished", make_version)
