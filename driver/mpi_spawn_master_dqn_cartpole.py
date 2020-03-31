#!/usr/bin/env python
from mpi4py import MPI
import numpy
import sys

# Number of workers to spawn from each process
n_workers = 4

# Spawn 'n_workers' process from each process
child_comm = MPI.COMM_SELF.Spawn(sys.executable, args=['mpi_scripts/mpi_spawn_worker_dqn_cartpole.py'], maxprocs=n_workers)

# Get communicator size and process rank
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()

# Convert to an integer before broadcasting
rank = numpy.array(rank, 'i')

# Broadcast to children
child_comm.Bcast([rank, MPI.INT], root=MPI.ROOT)

print("I am parent " + str(rank) + "/" + str(size))

child_comm.Disconnect()


