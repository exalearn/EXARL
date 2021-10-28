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
sys.path.insert(0, os.path.abspath('../..'))
sys.path.append('../../exarl/config')
sys.path.append('../../exarl/envs')
sys.path.append('../../exarl/envs/env_vault')
sys.path.append('../../exarl/envs/env_vault/ExaCartpoleStatic')
sys.path.append('../../exarl/envs/env_vault/ExaWaterClusterDiscrete')
sys.path.append('../../exarl/envs/env_vault/ExaCH')
sys.path.append('../../exarl/envs/env_vault/ExaCOVID')
sys.path.append('../../exarl/envs/env_vault/ExaBoosterDiscrete')
sys.path.append('../../exarl/workflows')
sys.path.append('../../exarl/workflows/registration')
sys.path.append('../../exarl/agents/agent_vault')
sys.path.append('../../exarl/agents/registration')
sys.path.append('../../exarl/agents/')
sys.path.append('../../exarl/agents/agent_vault/dqn')
sys.path.append('../../exarl/agents/agent_vault/ddpg_vtrace')
sys.path.append('../../exarl/agents/agent_vault/ddpg')
sys.path.append('../../exarl/base')
sys.path.append('../../exarl/base/workflow_base')
sys.path.append('../../exarl/base/test_state')
sys.path.append('../../exarl/base/test_action')
sys.path.append('../../exarl/base/state')
sys.path.append('../../exarl/base/learner_base')
sys.path.append('../../exarl/base/env_base')
sys.path.append('../../exarl/base/dataset_base')
sys.path.append('../../exarl/base/data_exchange')
sys.path.append('../../exarl/base/agent_base')
sys.path.append('../../exarl/base/action')
sys.path.append('../../exarl/utils/OUActionNoise')
sys.path.append('../../exarl/utils/analyze_reward')
sys.path.append('../../exarl/utils/introspect')
sys.path.append('../../exarl/utils/profile')
sys.path.append('../../exarl/utils/typing')
sys.path.append('../../exarl/utils')
sys.path.append('../../exarl/config/learner_cfg.json')

# -- Project information -----------------------------------------------------

project = 'EXARL'
copyright = '2021, ExaLearn Control Team'
author = 'ExaLearn Control Team'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.coverage',
              'sphinx.ext.doctest', 'sphinx.ext.intersphinx',
              'sphinx.ext.todo',  'sphinx.ext.autosummary',
              'sphinx.ext.ifconfig', 'sphinx.ext.viewcode',
              'sphinx.ext.inheritance_diagram']
# extensions = ['sphinx.ext.autodoc']

# Configuration of sphinx.ext.coverage
coverage_show_missing_items = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
