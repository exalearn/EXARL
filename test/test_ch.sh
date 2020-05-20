#!/bin/bash

python ../driver/driver_example.py

mpiexec -np 2 ../driver/driver_example.py

# TODO: need to setup config files for testing
