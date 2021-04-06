ECP Annual Meeting 21 - EXARL tutorial
======================================

Installation on NERSC CORI
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