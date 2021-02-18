# © (or copyright) 2020. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
# National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
# Department of Energy/National Nuclear Security Administration. All rights in the program are
# reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
# Security Administration. The Government is granted for itself and others acting on its behalf a
# nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
# derivative works, distribute copies to the public, perform publicly and display publicly, and
# to permit others to do so.

import sys
import os
import exarl as erl
from mpi4py import MPI

# Move to ExaBuffMPI
class ExaMPIBuff(erl.ExaData):
    def __init__(self, comm, rank, size=None, data=None):
        self.comm = comm
        self.rank = rank

        if data is not None:
            dataBytes = MPI.pickle.dumps(data)
            size = len(dataBytes)
        super().__init__(bytes, size)

        totalSize = 0
        if comm.rank == rank:
            totalSize = size * self.comm.size
        self.win = MPI.Win.Allocate(totalSize, disp_unit=size, comm=self.comm.raw())
        self.buff = bytearray(self.dataSize)
        # If we are given data to start lets put it in our buffer
        # Since everyone should call this everyone should get a start value!
        if data is not None:
            self.push(data)
            self.win.Fence(self.rank)

    def __del__(self):
        self.win.Free()

    def pop(self, rank, count=1):
        self.win.Lock(self.rank)
        self.win.Get_accumulate(
            self.buff, self.buff, self.rank, target=[rank, self.dataSize], op=MPI.NO_OP
        )
        self.win.Unlock(self.rank)
        return MPI.pickle.loads(self.buff)

    def push(self, data):
        toSend = MPI.pickle.dumps(data)
        # print(len(toSend), self.dataSize)
        assert len(toSend) == self.dataSize
        self.win.Lock(self.rank)
        self.win.Accumulate(
            toSend, self.rank, target=[self.comm.rank, self.dataSize], op=MPI.REPLACE
        )
        self.win.Unlock(self.rank)
