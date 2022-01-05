#!/usr/bin/env bash

set -e

# Get hostname of the machine
target=${EXARL_TARGET:-$(hostname --fqdn)}
echo "target = $target"

if [[ $target=*"summit"* ]]
then
    module purge
    module load cuda/11.0.3
    module load open-ce/1.4.0-py37-0
    module load git-lfs

    conda env create -f setup/environment.yml
    
elif [[ $target="darwin"* ]]
then
    module load openmpi/3.1.0-gcc_7.3.0
    module load anaconda/Anaconda3.2019.10
    module load cuda/10.1

    conda create -name exarl python=3.7
    pip install -e setup/ --user
else
    echo "Don't recognize this target/hostname: $(target)."
fi
