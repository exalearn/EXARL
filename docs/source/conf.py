# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath('../../exarl'))
# for x in os.walk('../../exarl'):
# sys.path.insert(0, x[0])

# sys.path.append(os.path.abspath('../../exarl'))
# sys.path.append(os.path.abspath('../../exarl/config'))
# sys.path.append(os.path.abspath('../../exarl/config/learner_cfg.json'))
# sys.path.append(os.path.abspath('../../exarl/envs'))
# sys.path.append(os.path.abspath('../../exarl/envs/env_vault'))
# sys.path.append(os.path.abspath('../../exarl/envs/env_vault/ExaCartpoleStatic'))
# sys.path.append(os.path.abspath('../../exarl/envs/env_vault/ExaWaterClusterDiscrete'))
# sys.path.append(os.path.abspath('../../exarl/envs/env_vault/ExaCH'))
# sys.path.append(os.path.abspath('../../exarl/envs/env_vault/ExaCOVID'))
# sys.path.append(os.path.abspath('../../exarl/envs/env_vault/ExaBoosterDiscrete'))
# sys.path.append(os.path.abspath('../../exarl/workflows'))
# sys.path.append(os.path.abspath('../../exarl/workflows/registration'))
# sys.path.append(os.path.abspath('../../exarl/workflows/workflow_vault'))
# sys.path.append(os.path.abspath('../../exarl/workflows/workflow_vault/async_learner'))
# sys.path.append(os.path.abspath('../../exarl/workflows/workflow_vault/rma_learner'))
# sys.path.append(os.path.abspath('../../exarl/workflows/workflow_vault/sync_learner'))
# sys.path.append(os.path.abspath('../../exarl/workflows/workflow_vault/tester_learner'))
# sys.path.append(os.path.abspath('../../exarl/agents/'))
# sys.path.append(os.path.abspath('../../exarl/agents/agent_vault'))
# sys.path.append(os.path.abspath('../../exarl/agents/agent_vault/dqn'))
# sys.path.append(os.path.abspath('../../exarl/agents/agent_vault/ddpg_vtrace'))
# sys.path.append(os.path.abspath('../../exarl/agents/agent_vault/ddpg'))
# sys.path.append(os.path.abspath('../../exarl/agents/registration'))
# sys.path.append(os.path.abspath('../../exarl/base'))
# sys.path.append(os.path.abspath('../../exarl/base/workflow_base'))
# sys.path.append(os.path.abspath('../../exarl/base/test_state'))
# sys.path.append(os.path.abspath('../../exarl/base/test_action'))
# sys.path.append(os.path.abspath('../../exarl/base/state'))
# sys.path.append(os.path.abspath('../../exarl/base/learner_base'))
# sys.path.append(os.path.abspath('../../exarl/base/env_base'))
# sys.path.append(os.path.abspath('../../exarl/base/dataset_base'))
# sys.path.append(os.path.abspath('../../exarl/base/data_exchange'))
# sys.path.append(os.path.abspath('../../exarl/base/comm_base'))
# sys.path.append(os.path.abspath('../../exarl/base/agent_base'))
# sys.path.append(os.path.abspath('../../exarl/base/action'))
# sys.path.append(os.path.abspath('../../exarl/utils/OUActionNoise'))
# sys.path.append(os.path.abspath('../../exarl/utils/analyze_reward'))
# sys.path.append(os.path.abspath('../../exarl/utils/introspect'))
# sys.path.append(os.path.abspath('../../exarl/utils/profile'))
# sys.path.append(os.path.abspath('../../exarl/utils/typing'))
# sys.path.append(os.path.abspath('../../exarl/utils'))
# sys.path.append(os.path.abspath('../../exarl/network'))
# sys.path.append('../../exarl/config/learner_cfg.json')

print(sys.path)

# -- Project information -----------------------------------------------------

project = 'EXARL'
copyright = '2021, ExaLearn Control Team'
author = 'ExaLearn Control Team'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    # leaving these here in case Josh needs them for debugging
    # 'sphinx.ext.autodoc',
    # 'sphinx.ext.autosummary',
    'autoapi.extension',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.inheritance_diagram'
]

# autoapi options here
autoapi_type = 'python'
autoapi_dirs = ['../../exarl']
autoapi_keep_files = True  # enable incremental build, keep files for examination

# autosummary triggers autodoc which forces code execution
autosummary_generate = False
# napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_ivar = True
napoleon_use_rtype = False

# Configuration of sphinx.ext.coverage
coverage_show_missing_items = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []  # ['__init__.py', '__main__.py']
# autodoc_mock_imports = ['exarl/agents/agent_vault']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'collapse_navigation': True,
    'display_version': True
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
