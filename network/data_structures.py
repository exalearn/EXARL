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
import numpy as np
from network.simple_comm import ExaSimple
MPI=ExaSimple.MPI

# Move to ExaBuffMPI
class ExaMPIBuff(erl.ExaData):
    def __init__(self, comm, rank, size=None, data=None, length=1, max_model_lag=None):
        self.comm = comm
        self.rank = rank

        if data is not None:
            dataBytes = MPI.pickle.dumps(data)
            size = len(dataBytes)
        super().__init__(bytes, size, comm_size=comm.size, max_model_lag=max_model_lag)

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


class ExaMPIStack(erl.ExaData):
    def __init__(self, comm, rank, size=None, data=None, length=32, max_model_lag=None):
        self.comm = comm
        self.rank = rank
        self.length = length

        if data is not None:
            dataBytes = MPI.pickle.dumps(data)
            size = len(dataBytes)
        super().__init__(bytes, size, comm_size=comm.size, max_model_lag=max_model_lag)
        self.buff = bytearray(self.dataSize)
        self.plus = np.array([1], dtype=np.int64)
        self.minus = np.array([-1], dtype=np.int64)

        totalSize = 0
        headSize = 0
        if comm.rank == rank:
            totalSize = size * self.length
            headSize = MPI.INT64_T.Get_size()

        self.head = []
        self.win = []
        for i in range(comm.size):
            # Setup head window
            self.head.append(MPI.Win.Allocate(headSize, comm=self.comm.raw()))
            self.head[i].Lock(self.rank)
            self.head[i].Accumulate(
                np.zeros(1, dtype=np.int64), self.rank, op=MPI.REPLACE
            )
            self.head[i].Unlock(self.rank)
            self.head[i].Fence(self.rank)

            # Setup data window
            self.win.append(
                MPI.Win.Allocate(totalSize, disp_unit=size, comm=self.comm.raw())
            )
            self.win[i].Fence(self.rank)

    def __del__(self):
        for i in range(self.comm.size):
            self.win[i].Free()
            self.head[i].Free()

    def pop(self, rank, count=1):
        ret = True
        head = np.zeros(1, dtype=np.int64)
        rank = int(rank)
        self.head[rank].Lock(self.rank)
        req = self.head[rank].Rget_accumulate(self.minus, head, self.rank, op=MPI.SUM)
        req.wait()

        if head[0] > 0:
            index = (head[0] - 1) % self.length

            self.win[rank].Lock(self.rank)
            self.win[rank].Get_accumulate(
                self.buff,
                self.buff,
                self.rank,
                target=[index, self.dataSize],
                op=MPI.NO_OP,
            )
            self.win[rank].Unlock(self.rank)

        if head[0] <= 0:
            self.head[rank].Accumulate(
                self.plus, self.rank, op=MPI.SUM
            )
            if head[0] == 0:
                ret = False

        self.head[rank].Unlock(self.rank)

        if ret:
            return MPI.pickle.loads(self.buff)
        return None

    def push(self, data):
        rank = self.comm.rank
        toSend = MPI.pickle.dumps(data)
        assert len(toSend) == self.dataSize

        head = np.zeros(1, dtype=np.int64)

        self.head[rank].Lock(self.rank)
        # If we don't wait we can't guarentee the value until after the lock...
        req = self.head[rank].Rget_accumulate(self.plus, head, self.rank, op=MPI.SUM)
        req.wait()
        index = head[0] % self.length

        self.win[rank].Lock(self.rank)
        self.win[rank].Accumulate(
            toSend, self.rank, target=[index, self.dataSize], op=MPI.REPLACE
        )
        self.win[rank].Unlock(self.rank)

        self.head[rank].Unlock(self.rank)

class ExaMPIQueue(erl.ExaData):
    def __init__(self, comm, rank, size=None, data=None, length=32, max_model_lag=None):
        self.comm = comm
        self.rank = rank
        self.length = length

        if data is not None:
            dataBytes = MPI.pickle.dumps(data)
            size = len(dataBytes)
        super().__init__(bytes, size, comm_size=comm.size, max_model_lag=max_model_lag)
        self.buff = bytearray(self.dataSize)
        self.plus = np.array([1], dtype=np.int64)
        self.minus = np.array([-1], dtype=np.int64)

        totalSize = 0
        headSize = 0
        tailSize = 0
        if comm.rank == rank:
            totalSize = size * self.length
            headSize = MPI.INT64_T.Get_size()
            tailSize = MPI.INT64_T.Get_size()

        self.head = []
        self.tail = []
        self.win = []
        for i in range(comm.size):
            # Setup head window
            self.head.append(MPI.Win.Allocate(headSize, comm=self.comm.raw()))
            self.head[i].Lock(self.rank)
            self.head[i].Accumulate(
                np.zeros(1, dtype=np.int64), self.rank, op=MPI.REPLACE
            )
            self.head[i].Unlock(self.rank)
            self.head[i].Fence(self.rank)

            # Setup tail window
            self.tail.append(MPI.Win.Allocate(headSize, comm=self.comm.raw()))
            self.tail[i].Lock(self.rank)
            self.tail[i].Accumulate(
                np.zeros(1, dtype=np.int64), self.rank, op=MPI.REPLACE
            )
            self.tail[i].Unlock(self.rank)
            self.tail[i].Fence(self.rank)

            # Setup data window
            self.win.append(
                MPI.Win.Allocate(totalSize, disp_unit=size, comm=self.comm.raw())
            )
            self.win[i].Fence(self.rank)

    def __del__(self):
        for i in range(self.comm.size):
            self.win[i].Free()
            self.head[i].Free()

    def pop(self, rank, count=1):
        ret = True
        head = np.zeros(1, dtype=np.int64)
        tail = np.zeros(1, dtype=np.int64)
        rank = int(rank)

        self.head[rank].Lock(self.rank)
        self.tail[rank].Lock(self.rank)

        # Read the head and tail pointers. Don't modify (MPI.NO_OP)
        reqHead = self.head[rank].Rget_accumulate(self.minus, head, self.rank, op=MPI.NO_OP)
        reqTail = self.tail[rank].Rget_accumulate(self.minus, tail, self.rank, op=MPI.NO_OP)
        reqHead.wait()
        reqTail.wait()

        # Is there space
        if head[0] > tail[0]:
            index = tail[0] % self.length
            self.win[rank].Lock(self.rank)
            self.win[rank].Get_accumulate(
                self.buff,
                self.buff,
                self.rank,
                target=[index, self.dataSize],
                op=MPI.NO_OP,
            )
            self.win[rank].Unlock(self.rank)

            # Inc the tail pointer
            self.tail[rank].Accumulate(self.plus, self.rank, op=MPI.SUM)
        else:
            ret = False

        self.tail[rank].Unlock(self.rank)
        self.head[rank].Unlock(self.rank)

        if ret:
            return MPI.pickle.loads(self.buff)
        return None

    def push(self, data):
        rank = self.comm.rank
        toSend = MPI.pickle.dumps(data)
        assert len(toSend) <= self.dataSize

        head = np.zeros(1, dtype=np.int64)
        tail = np.zeros(1, dtype=np.int64)

        self.head[rank].Lock(self.rank)
        self.tail[rank].Lock(self.rank)

        # If we don't wait we can't guarentee the value until after the lock...
        reqHead = self.head[rank].Rget_accumulate(self.plus, head, self.rank, op=MPI.SUM)
        reqTail = self.tail[rank].Rget_accumulate(self.plus, tail, self.rank, op=MPI.NO_OP)
        reqHead.wait()
        reqTail.wait()

        index = head[0] % self.length
        self.win[rank].Lock(self.rank)
        self.win[rank].Accumulate(
            toSend, self.rank, target=[index, self.dataSize], op=MPI.REPLACE
        )
        self.win[rank].Unlock(self.rank)

        if head + 1 == tail:
            self.tail[rank].Accumulate(
                np.array([1], dtype=np.int64), self.rank, op=MPI.SUM
            )

        self.tail[rank].Unlock(self.rank)
        self.head[rank].Unlock(self.rank)