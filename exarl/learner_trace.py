import sys
import os
import numpy as np
from exarl.simple_comm import ExaSimple 
from mpi4py import MPI

class learner_trace:
    def __init__(self, comm):
        self.comm = comm
        self.raw = comm.raw()
        self.trace = None if comm.rank else []
        self.counters = None if comm.rank else np.zeros(comm.size, dtype=np.int64)
        self.win = MPI.Win.Create(self.counters, disp_unit=MPI.INT64_T.Get_size(), comm=self.raw)
        self.win.Fence()

    def update(self, index=None):
        if index == None:
            index = self.comm.rank
        data = np.ones(1, dtype=np.int64)
        self.win.Accumulate(data, 0, target=[index,1])

    def snapshot(self, reward):
        if self.comm.rank == 0:
            data = np.zeros(self.comm.size, dtype=np.int64)
            self.win.Get_accumulate(data, data, 0, op=MPI.NO_OP)
            self.trace.append((data, reward))
            return data

    def write(self):
        self.win.Fence()
        if self.comm.rank == 0:
            for i in self.trace:
                print(i)
