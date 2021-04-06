ECP Annual Meeting 21 - EXARL Tutorial
======================================

Getting Training Account on NERSC
---------------------------------
- Go to https://iris.nersc.gov/train 
- Enter the randomly-generated 4-letter code for this tutorial - `dBXa`
- Fill out the rest of the form appropriately

This will create a temporary account for the user, with the username `trainXXX` where `XXX` is a 3-digit number. 
They will be members of the project `ntrain`. NERSC has already given read access to members of `ntrain` to the
relevant data sets for this tutorial in the directory `/global/cfs/cdirs/m3363/exarl_data`.
To access the cori machine, you need to download google authenticator app/authy. Refer https://docs.nersc.gov/connect/mfa/.

.. code-block:: bash

    $ ssh -XY username@cori.nersc.gov. 

when prompted, please type your password+ssh token from google authenticator. No spaces.


Installation on NERSC Cori
--------------------------
- Go to the scratch directory

.. code-block:: bash

    $ cd $SCRATCH

- Create a directory for tutorial and enter the directory

.. code-block:: bash

    $ mkdir EXARL_tutorial
    $ cd EXARL_tutorial

- Clone the EXARL repository

.. code-block:: bash

    $ git clone https://github.com/exalearn/ExaRL.git
    $ cd ExaRL

- Start Shifter image

.. code-block:: bash

    $ shifter --image=registry.nersc.gov/apg/exarl-ngc:0.1 /bin/bash

- Install dependencies

.. code-block:: bash

    $ pip install -e .

- Exit out of the Shifter image

.. code-block:: bash

    $ exit

- Go to the top level and create a directory for your requirements

.. code-block:: bash

    $ cd ..
    $ mkdir Runs
    $ cd Runs

- Submit job

.. code-block:: bash

    $ sbatch scripts/cori_tutorial_cpu.sh

- For training on GPUs

.. code-block:: bash

    $ module load cgpu
    $ sbatch scripts/cori_tutorial_cpu.sh