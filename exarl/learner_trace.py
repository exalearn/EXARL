import sys
import os
import numpy as np
from exarl.simple_comm import ExaSimple 
from mpi4py import MPI

class learner_trace:
    def __init__(self, comm):
        self.comm = comm
        self.raw = comm.raw()

        self.trace = None
        self.counters = None
        self.rewards = None
        if comm.rank == 0:
            self.trace = []
            self.counters = np.zeros(comm.size, dtype=np.int64)
            self.rewards = np.zeros(comm.size, dtype=np.float64)

        self.winCounter = MPI.Win.Create(self.counters, disp_unit=MPI.INT64_T.Get_size(), comm=self.raw)
        self.winReward = MPI.Win.Create(self.rewards, disp_unit=MPI.DOUBLE.Get_size(), comm=self.raw)

        self.winCounter.Fence()
        self.winReward.Fence()

    def update(self, reward):
        data = np.ones(1, dtype=np.int64)
        self.winCounter.Accumulate(data, 0, target=[self.comm.rank,1])

        data = np.array([reward],  dtype=np.float64)
        self.winReward.Accumulate(data, 0, target=[self.comm.rank,1], op=MPI.REPLACE)

    def snapshot(self):
        if self.comm.rank == 0:
            counts = np.zeros(self.comm.size, dtype=np.int64)
            self.winCounter.Get_accumulate(counts, counts, 0, op=MPI.NO_OP)

            rewards = np.zeros(self.comm.size, dtype=np.float64)
            self.winReward.Get_accumulate(rewards, rewards, 0, op=MPI.NO_OP)
            
            data = (counts, rewards)
            self.trace.append(data)
            return data

    def write(self):
        self.winCounter.Fence()
        self.winReward.Fence()
        if self.comm.rank == 0:
            with open("tr_" + str(self.comm.size) + ".txt", "w") as f:
                for count, reward in self.trace:
                    line = ",".join([str(x) for x in count]) + "," + ",".join([str(x) for x in reward]) + "\n"
                    f.write(line)
            
