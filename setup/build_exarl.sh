#!/usr/bin/env bash

set -e

# Get hostname of the machine
target=${EXARL_TARGET:-$(hostname --fqdn)}
echo "target = $target"

if [[ $target == *"summit"* ]]; then
    module purge
    module load cuda/11.0.3
    module load open-ce/1.4.0-py37-0
    module load git-lfs
    conda env create -f setup/summit_environment.yml
elif [[ $target == "darwin"* ]]; then
    module load openmpi/3.1.6-gcc_9.4.0
    module load miniconda3/py37_4.9.2
    module load cuda/11.4.2
    conda create --name exarl_darwin
    source activate exarl_darwin
    pip install -e setup/ --user
else
    echo "Don't recognize this target/hostname: $(target)."
    PYTHON_VERSION=3.7
    # Use the miniconda installer for setup of conda itself
    pushd .
    cd
    mkdir -p download
    cd download
    if [[ ! -f miniconda.sh ]]; then
        wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
            -O miniconda.sh
    fi
    chmod +x miniconda.sh && ./miniconda.sh -b -f
    conda update --yes conda
    echo "Creating conda environment."
    conda create -n exarl_$target --yes python=$PYTHON_VERSION
    cd ..
    popd
    # Activate the python environment
    source activate exarl_$target
    pip install -e setup/ --user
fi
