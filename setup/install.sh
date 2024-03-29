#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

set -e

echo 'List files from cached directories'
if [ -d $HOME/download ]; then
    echo 'download:'
    ls $HOME/download
fi
if [ -d $HOME/.cache/pip ]; then
    echo 'pip:'
    ls $HOME/.cache/pip
fi

# Deactivate the travis-provided virtual environment and setup a
# conda-based environment instead
deactivate

# Add the miniconda bin directory to $PATH
export PATH=/home/travis/miniconda3/bin:$PATH
echo $PATH
export PYTHON_VERSION=3.7

# Use the miniconda installer for setup of conda itself
pushd .
cd
mkdir -p download
cd download
if [[ ! -f /home/travis/miniconda3/bin/activate ]]
then
    if [[ ! -f miniconda.sh ]]
    then
        wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
             -O miniconda.sh
    fi
    chmod +x miniconda.sh && ./miniconda.sh -b -f
    conda update --yes conda
    echo "Creating environment to run tests in."
    conda create -n testenv --yes python=$PYTHON_VERSION
fi
cd ..
popd

# Activate the python environment we created.
source activate testenv

# Install requirements via pip in our conda environment
pip install -r setup/requirements.txt

# Install the following only if running tests
if [[ "$SKIP_TESTS" != "true" ]]; then
    conda install --yes pytorch torchvision -c pytorch
    conda install --yes tensorflow                                                                                                                                                     
    conda install --yes pandas                                                                                 
    conda install --yes numba
    conda install --yes mpi4py
    conda install --yes plotly
    # Installation
    # python setup.py install
fi
