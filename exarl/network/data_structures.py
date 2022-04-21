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
import numpy as np
from exarl.base import ExaData
from exarl.base.comm_base import ExaComm
from exarl.network.typing import TypeUtils
from exarl.utils.introspect import introspectTrace

class ExaMPIConstant:
    """
    This class is built to maintain a single value using mpi rdma.
    Each rank will have a window the size of the type.

    Attributes
    ----------
    comm : mpi4py.MPI.Comm
        raw MPI communicator

    npType : type
        numpy type of constant

    mpiType : type
        mpi type of the constant

    rank : int
        rank that hosts the data

    win : MPI.win
        MPI window for constant
    sum : int
        internal constant numpy 1 for incrementing

    buff : numpy array
        internal numpy buffer used for RMA ops

    name : string
        name of the constant for debugging

    """

    def __init__(self, comm, rank_mask, the_type, name=None):
        """
        Parameters
        ----------
        MPI : mpi4py.MPI
            mpi4py's MPI access point

        comm : mpi4py.MPI.Comm
            Communicator for all ranks involved

        rank_mask : int, optional
            host of the window

        the_type : int, optional
            python type (int, float)

        name : string, optional
            name of constant for debbuging
        """
        self.MPI = ExaComm.get_MPI()
        self.comm = comm.raw()
        self.npType = TypeUtils.np_type_converter(the_type, promote=True)
        self.mpiType = TypeUtils.mpi_type_converter(the_type, promote=True)
        self.size = self.mpiType.Get_size()
        data = None
        if rank_mask:
            self.rank = self.comm.rank
            data = np.zeros(1, dtype=self.npType)
        self.win = self.MPI.Win.Create(data, self.size, comm=self.comm)
        self.sum = np.ones(1, dtype=self.npType)
        self.buff = np.zeros(1, dtype=self.npType)
        self.name = name

    @introspectTrace(name=True)
    def put(self, value, rank):
        """
        Places a constant on a given rank

        Parameters
        ----------
        value: int
            Number to send to all ranks

        rank: integer
            Host rank of the actual number

        """
        data = np.array(value, dtype=self.npType)
        self.win.Lock(rank)
        self.win.Accumulate(data, target_rank=rank, op=self.MPI.REPLACE)
        self.win.Unlock(rank)

    @introspectTrace(name=True)
    def get(self, rank):
        """
        Gets a constant from a given rank

        Parameters
        ----------

        rank : integer
            Host rank of the actual number

        Returns
        -------
        int
            Constant from host rank
        """
        self.win.Lock(rank)
        self.win.Get_accumulate(self.sum, self.buff, target_rank=rank, op=self.MPI.NO_OP)
        self.win.Unlock(rank)
        return self.buff[0]

    @introspectTrace(name=True)
    def inc(self, rank):
        """
        Increments a constant on host rank

        Parameters
        ----------

        rank : integer
            Host rank of the actual number

        Returns
        -------
        int
            Constant from host rank before the increment
        """
        self.win.Lock(rank)
        self.win.Get_accumulate(self.sum, self.buff, target_rank=rank, op=self.MPI.SUM)
        self.win.Unlock(rank)
        return self.buff[0]

    @introspectTrace(name=True)
    def min(self, value, rank):
        """
        Takes the min of new value and constant on host rank

        Parameters
        ----------
        value : integer
            To value to compare constant with

        rank : integer
            Host rank of the actual number

        Returns
        -------
        int
            Minimum of the new value and constant
        """
        data = np.array(value, dtype=self.npType)
        self.win.Lock(rank)
        self.win.Get_accumulate(data, self.buff, target_rank=rank, op=self.MPI.MIN)
        self.win.Unlock(rank)
        return min(self.buff[0], value)

class ExaMPIBuffUnchecked(ExaData):
    """
    This class is creates an RMA buffer of a fixed size on each rank.
    The buffer is used to send and recieve data across all participating ranks.
    This buffer does not check to see if it is overwriting data or if there is
    valid data from a get.  This class always succeds a pop.

    Attributes
    ----------
    comm : mpi4py.MPI.Comm
        raw MPI communicator

    win : MPI.win
        MPI window for buffer

    buff : bytearray
        internal buffer used for RMA ops


    **Intializer**

    Parameters
    ----------
    MPI : mpi4py.MPI
            mpi4py's MPI access point
    comm : MPI Comm
        Communicator for all ranks involved
    data : list
        Example data used to create buffer
    rank_mask : int, optional
        host of the window
    length : int, optional
        Not used
    max_model_lag : int, optional
        Not used
    failPush : bool, optional
        Not used
    name : string, optional
        name of constant for debbuging
    """
    def __init__(self, comm, data, rank_mask=None, length=1, max_model_lag=None, failPush=False, name=None):
        """
        Parameters
        ----------
        comm : MPI Comm
            Communicator for all ranks involved
        data : list
            Example data used to create buffer
        rank_mask : int, optional
            host of the window
        length : int, optional
            Not used
        max_model_lag : int, optional
            Not used
        failPush : bool, optional
            Not used
        name : string, optional
            name of constant for debbuging
        """
        self.MPI = ExaComm.get_MPI()
        self.comm = comm
        self.length = 1
        dataBytes = self.MPI.pickle.dumps(data)
        # JS: adding one because pickle is dumb
        size = len(dataBytes) + 1

        super().__init__(bytes, size, comm_size=comm.size, max_model_lag=None, name=name)

        totalSize = 0
        if rank_mask:
            totalSize = size
        self.win = self.MPI.Win.Allocate(totalSize, disp_unit=1, comm=self.comm.raw())
        self.buff = bytearray(self.dataSize)

        # If we are given data to start lets put it in our buffer
        # Since everyone should call this everyone should get a start value!
        if rank_mask:
            self.push(data)

    def __del__(self):
        self.win.Free()

    @introspectTrace(name=True)
    def pop(self, rank, count=1):
        """
        Returns value of buffer at given rank.  There is no check
        done to see if the data is valid.

        Parameters
        ----------
        rank : integer
            Host rank where to take data from

        count : integer
            How many pops to perform

        Returns
        -------
        list
            Buffer at given rank
        """
        self.win.Lock(rank)
        self.win.Get_accumulate(
            self.buff,
            self.buff,
            rank,
            target=[0, self.dataSize],
            op=self.MPI.NO_OP,
        )
        self.win.Unlock(rank)
        return self.MPI.pickle.loads(self.buff)

    @introspectTrace(name=True)
    def push(self, data, rank=None):
        """
        Pushes data to a rank's buffer.

        Parameters
        ----------
        data : list
            Data to be pushed to rank's buffer

        rank : integer
            Host rank of the actual number

        Returns
        -------
        list
            Returns a capacity of 1 and loss of 1
        """
        if rank is None:
            rank = self.comm.rank

        toSend = self.MPI.pickle.dumps(data)
        assert len(toSend) <= self.dataSize, str(len(toSend)) + " vs " + str(self.dataSize)

        self.win.Lock(rank)
        # Accumulate is element-wise atomic vs put which is not
        self.win.Accumulate(
            toSend, rank, target=[0, len(toSend)], op=self.MPI.REPLACE
        )
        self.win.Unlock(rank)
        return 1, 1

class ExaMPIBuffChecked(ExaData):
    """
    This class is creates an RMA buffer of a fixed size on each rank.
    The buffer is used to send and recieve data across all participating ranks.
    On pop, checks to see if the data is first valid.

    Attributes
    ----------
    MPI : mpi4py.MPI
            mpi4py's MPI access point

    comm : mpi4py.MPI.Comm
        raw MPI communicator

    win : MPI.win
        MPI window for buffer

    buff : bytearray
        internal buffer used for RMA ops

    Methods
    -------
    pop(value, rank, count)
        Returns value stored in buffer at rank

    push(self, data, rank)
        Pushes data to buffer at rank

    """
    def __init__(self, comm, data, rank_mask=None, length=1, max_model_lag=None, failPush=False, name=None):
        """
        Parameters
        ----------
        comm : mpi4py.MPI.Comm
            Communicator for all ranks involved
        data : list
            Example data used to create buffer
        rank_mask : int, optional
            host of the window
        length : int
            Not used
        max_model_lag : int
            Not used
        failPush : bool
            Not used
        name : string, optional
            name of constant for debbuging
        """
        self.MPI = ExaComm.get_MPI()
        self.comm = comm
        self.length = 1

        self.dataBytes = bytearray(self.MPI.pickle.dumps((data, np.int64(0))))
        size = len(self.dataBytes)

        super().__init__(bytes, size, comm_size=comm.size, max_model_lag=None, name=name)

        totalSize = 0
        if rank_mask:
            totalSize = size
        self.win = self.MPI.Win.Allocate(totalSize, disp_unit=1, comm=self.comm.raw())
        self.buff = bytearray(self.dataSize)

        if rank_mask:
            self.win.Lock(self.comm.rank)
            self.win.Accumulate(
                self.dataBytes, self.comm.rank, target=[0, self.dataSize], op=self.MPI.REPLACE
            )
            self.win.Unlock(self.comm.rank)

    def __del__(self):
        self.win.Free()

    @introspectTrace(name=True)
    def pop(self, rank, count=1):
        """
        Returns value of buffer at given rank.
        Checks to see if the data is valid first.

        Parameters
        ----------
        rank : integer
            Host rank where to take data from

        count : integer, optional
            How many pops to perform

        Returns
        -------
        list
            Buffer at given rank if valid
        """
        self.win.Lock(rank)
        self.win.Get_accumulate(
            self.dataBytes,
            self.buff,
            rank,
            target=[0, self.dataSize],
            op=self.MPI.REPLACE
        )

        self.win.Unlock(rank)
        data, valid = self.MPI.pickle.loads(self.buff)
        if valid:
            return data
        return None

    @introspectTrace(name=True)
    def push(self, data, rank=None):
        """
        Pushes data to a rank's buffer.

        Parameters
        ----------
        data : list
            Data to be pushed to rank's buffer

        rank : integer, optional
            Host rank of the actual number

        Returns
        -------
        list
            Returns a capacity of 1 and loss if data is overwritten
        """
        if rank is None:
            rank = self.comm.rank

        toSend = bytearray(self.MPI.pickle.dumps((data, np.int64(1))))
        assert len(toSend) <= self.dataSize

        self.win.Lock(rank)
        self.win.Get_accumulate(
            toSend,
            self.buff,
            rank,
            target=[0, self.dataSize],
            op=self.MPI.REPLACE
        )
        self.win.Unlock(rank)
        _, valid = self.MPI.pickle.loads(self.buff)
        return 1, valid == 1

class ExaMPIDistributedQueue(ExaData):
    """
    This class creates a circular buffer in an RMA window across nodes in a communicator.
    Only one RMA window is made of length entries, thus there is only one host.

    Attributes
    ----------
    MPI : mpi4py.MPI
            mpi4py's MPI access point

    comm : mpi4py.MPI.Comm
        raw MPI communicator

    length : int
        capacity of the queue

    failPush : bool
        flag setting if push can overwrite data

    buff : bytearray
        internal buffer for queue used for RMA ops

    plus : np.array
        numpy constant for adding

    minus : np.array
        numpy constant for subtracting

    headBuffer : np.array
        buffer containing head counter

    tailBuffer : np.array
        buffer containing tail counter

    head : MPI.win
        RMA window based on headBuffer

    tail : MPI.win
        RMA window based on tailBuffer

    win : MPI.win
        MPI window based on buffer for queue

    """
    def __init__(self, comm, data=None, rank_mask=None, length=32, max_model_lag=None, failPush=False, name=None):
        """
        Parameters
        ----------
        comm : mpi4py.MPI.Comm
            Communicator for all ranks involved
        data : list, optional
            Example data used to create buffer
        rank_mask : int, optional
            host of the window
        length : int, optional
            capacity of queue
        max_model_lag : int, optional
            Will not consider data past given model valide
        failPush : bool, optional
            Fail to overwrite data if queue is full
        name : string, optional
            name of constant for debbuging
        """
        self.MPI = ExaComm.get_MPI()
        self.comm = comm
        self.length = length
        # This lets us fail a push when at full capacity
        # Otherwise will overwrite the oldest data
        self.failPush = failPush

        dataBytes = self.MPI.pickle.dumps(data)
        size = len(dataBytes) + 1

        super().__init__(bytes, size, comm_size=comm.size, max_model_lag=max_model_lag, name=name)
        self.buff = bytearray(self.dataSize)
        self.plus = np.array([1], dtype=np.int64)
        self.minus = np.array([-1], dtype=np.int64)

        totalSize = 0
        self.headBuff = None
        self.tailBuff = None
        disp = self.MPI.DOUBLE.Get_size()
        if rank_mask:
            totalSize = size * self.length
            self.headBuff = np.zeros(1, dtype=np.int64)
            self.tailBuff = np.zeros(1, dtype=np.int64)

        # Setup head window
        self.head = self.MPI.Win.Create(self.headBuff, disp, comm=self.comm.raw())

        # Setup tail window
        self.tail = self.MPI.Win.Create(self.tailBuff, disp, comm=self.comm.raw())

        # Setup data window
        self.win = self.MPI.Win.Allocate(totalSize, disp_unit=size, comm=self.comm.raw())

    def __del__(self):
        self.win.Free()

    @introspectTrace(name=True)
    def pop(self, rank, count=1):
        """
        Returns data from head of queue if there is data.

        Parameters
        ----------
        rank : integer
            Host rank where to take data from

        count : integer, optional
            How many pops to perform

        Returns
        -------
        list
            Data from queue if there is any.
        """
        ret = True
        head = np.zeros(1, dtype=np.int64)
        tail = np.zeros(1, dtype=np.int64)
        rank = int(rank)

        self.head.Lock(rank)
        self.tail.Lock(rank)

        # Read the head and tail pointers.
        reqHead = self.head.Rget_accumulate(self.minus, head, rank, op=self.MPI.NO_OP)
        reqTail = self.tail.Rget_accumulate(self.plus, tail, rank, op=self.MPI.SUM)
        reqHead.wait()
        reqTail.wait()

        # Is there space
        if head[0] > tail[0]:
            index = tail[0] % self.length
            self.win.Lock(rank)
            self.win.Get_accumulate(
                self.buff,
                self.buff,
                rank,
                target=[index, self.dataSize],
                op=self.MPI.NO_OP,
            )
            self.win.Unlock(rank)
        else:
            # Dec the tail pointer
            self.tail.Accumulate(self.minus, rank, op=self.MPI.SUM)
            ret = False

        self.tail.Unlock(rank)
        self.head.Unlock(rank)

        if ret:
            return self.MPI.pickle.loads(self.buff)
        return None

    @introspectTrace(name=True)
    def push(self, data, rank=None):
        """
        Pushes data to a rank's queue.

        Parameters
        ----------
        data : list
            Data to be pushed to rank's queue

        rank : integer, optional
            Rank to push data to

        Returns
        -------
        list
            Returns a capacity of queue and loss if data is overwritten
        """
        if rank is None:
            rank = self.comm.rank
        toSend = self.MPI.pickle.dumps(data)
        assert len(toSend) <= self.dataSize

        head = np.zeros(1, dtype=np.int64)
        tail = np.zeros(1, dtype=np.int64)

        self.head.Lock(rank)
        self.tail.Lock(rank)
        reqHead = self.head.Rget_accumulate(self.plus, head, rank, op=self.MPI.SUM)
        reqTail = self.tail.Rget_accumulate(self.plus, tail, rank, op=self.MPI.NO_OP)
        reqHead.wait()
        reqTail.wait()

        write = True
        headIndex = head[0] % self.length
        tailIndex = tail[0] % self.length

        if head[0] > tail[0] and headIndex == tailIndex:
            if self.failPush:
                write = False
                self.head.Accumulate(
                    self.minus, rank, op=self.MPI.SUM
                )
            else:
                self.tail.Accumulate(
                    self.plus, rank, op=self.MPI.SUM
                )
            lost = 1
            capacity = self.length
        else:
            lost = 0
            capacity = head[0] - tail[0] + 1

        if write:
            self.win.Lock(rank)
            self.win.Accumulate(
                toSend, rank, target=[headIndex, len(toSend)], op=self.MPI.REPLACE
            )
            self.win.Unlock(rank)

        self.tail.Unlock(rank)
        self.head.Unlock(rank)

        return capacity, lost

class ExaMPIDistributedStack(ExaData):
    """
    This class creates a stack in an RMA window across nodes in a communicator.
    Only one window is made, thus there is only one host.

    Attributes
    ----------
    MPI : mpi4py.MPI
            mpi4py's MPI access point

    comm : mpi4py.MPI.Comm
        raw MPI communicator

    length : int
        capacity of the stack

    failPush : bool
        flag setting if push can overwrite data

    buff : bytearray
        internal numpy buffer for stack used for RMA ops

    plus : np.array
        numpy constant for adding

    minus : np.array
        numpy constant for subtracting

    headBuffer : np.array
        buffer containing head counter

    tailBuffer : np.array
        buffer containing tail counter

    head : MPI.win
        window based on headBuffer

    tail : MPI.win
        window based on tailBuffer

    win : MPI.win
        MPI window based on buffer for stack

    """
    def __init__(self, comm, data, rank_mask=None, length=32, max_model_lag=None, failPush=False, name=None):
        """
        Parameters
        ----------
        comm : mpi4py.MPI.Comm
            Communicator for all ranks involved
        data : list
            Example data used to create buffer
        rank_mask : int, optional
            host of the window
        length : int, optional
            capacity of stack
        max_model_lag : int
            Will not consider data past given model valide
        failPush : bool, optional
            Fail to overwrite data if queue is full
        name : string, optional
            name of constant for debbuging
        """
        self.MPI = ExaComm.get_MPI()
        self.comm = comm
        self.length = length
        # This lets us fail a push when at full capacity
        # Otherwise will overwrite the oldest data
        self.failPush = failPush

        dataBytes = self.MPI.pickle.dumps(data)
        size = len(dataBytes) + 1
        super().__init__(bytes, size, comm_size=comm.size, max_model_lag=max_model_lag, name=name)

        self.buff = bytearray(self.dataSize)
        self.plus = np.array([1], dtype=np.int64)
        self.minus = np.array([-1], dtype=np.int64)

        totalSize = 0
        self.headBuff = None
        self.tailBuff = None
        disp = self.MPI.DOUBLE.Get_size()
        if rank_mask:
            totalSize = size * self.length
            self.headBuff = np.zeros(1, dtype=np.int64)
            self.tailBuff = np.zeros(1, dtype=np.int64)

        # Setup head window
        self.head = self.MPI.Win.Create(self.headBuff, disp, comm=self.comm.raw())

        # Setup tail window
        self.tail = self.MPI.Win.Create(self.tailBuff, disp, comm=self.comm.raw())

        # Setup data window
        self.win = self.MPI.Win.Allocate(totalSize, disp_unit=size, comm=self.comm.raw())

    def __del__(self):
        self.win.Free()

    @introspectTrace(name=True)
    def pop(self, rank, count=1):
        """
        Returns data from head of stack if there is data.

        Parameters
        ----------
        rank : integer
            Host rank where to take data from

        count : integer, optional
            How many pops to perform

        Returns
        -------
        list
            Data from stack if there is any.
        """
        ret = False
        head = np.zeros(1, dtype=np.int64)
        tail = np.zeros(1, dtype=np.int64)
        rank = int(rank)

        self.head.Lock(rank)
        self.tail.Lock(rank)

        # Read the head and tail pointers.
        reqHead = self.head.Rget_accumulate(self.minus, head, rank, op=self.MPI.SUM)
        reqTail = self.tail.Rget_accumulate(self.minus, tail, rank, op=self.MPI.NO_OP)
        reqHead.wait()
        reqTail.wait()
        # print("InPop", head[0], tail[0])
        if head[0] > tail[0]:
            ret = True
            index = (head[0] - 1) % self.length

            self.win.Lock(rank)
            self.win.Get_accumulate(
                self.buff,
                self.buff,
                rank,
                target=[index, self.dataSize],
                op=self.MPI.NO_OP,
            )
            self.win.Unlock(rank)

        else:
            self.head.Accumulate(
                self.plus, rank, op=self.MPI.SUM
            )

        self.tail.Unlock(rank)
        self.head.Unlock(rank)

        if ret:
            return self.MPI.pickle.loads(self.buff)
        return None

    @introspectTrace(name=True)
    def push(self, data, rank=None):
        """
        Pushes data to a rank's stack.

        Parameters
        ----------
        data : list
            Data to be pushed to rank's stack

        rank : integer, optional
            Host to push data to

        Returns
        -------
        list
            Returns a capacity of stack and loss if data is overwritten
        """
        if rank is None:
            rank = self.comm.rank
        toSend = self.MPI.pickle.dumps(data)
        assert len(toSend) <= self.dataSize

        head = np.zeros(1, dtype=np.int64)
        tail = np.zeros(1, dtype=np.int64)
        rank = int(rank)

        self.head.Lock(rank)
        self.tail.Lock(rank)

        # Read the head and tail pointers.
        reqHead = self.head.Rget_accumulate(self.plus, head, rank, op=self.MPI.SUM)
        reqTail = self.tail.Rget_accumulate(self.plus, tail, rank, op=self.MPI.NO_OP)
        reqHead.wait()
        reqTail.wait()

        # This is if we are going to loose data because we exceded capacity
        write = True
        if tail[0] + self.length == head[0]:
            if self.failPush:
                write = False
                self.head.Accumulate(
                    self.minus, rank, op=self.MPI.SUM
                )
            else:
                self.tail.Accumulate(
                    self.plus, rank, op=self.MPI.SUM
                )
            lost = 1
            capacity = self.length
        else:
            lost = 0
            capacity = head[0] - tail[0] + 1

        if write:
            # Actual write data
            index = head[0] % self.length
            self.win.Lock(rank)
            self.win.Accumulate(
                toSend, rank, target=[index, self.dataSize], op=self.MPI.REPLACE
            )
            self.win.Unlock(rank)

        self.tail.Unlock(rank)
        self.head.Unlock(rank)
        return capacity, lost

class ExaMPICentralizedStack(ExaData):
    """
    This class creates a stack in RMA windows across nodes in a communicator.
    There is a stack per rank.  Each rank acts as a host.

    Attributes
    ----------
    MPI : mpi4py.MPI
            mpi4py's MPI access point

    comm : mpi4py.MPI.Comm
        raw MPI communicator

    length : int
        capacity of the stack

    failPush : bool
        flag setting if push can overwrite data

    buff : bytearray
        internal buffer for stack used for RMA ops

    plus : np.array
        numpy constant for adding

    minus : np.array
        numpy constant for subtracting

    headBuffer : np.array
        buffer containing head counter

    tailBuffer : np.array
        buffer containing tail counter

    head : MPI.win
        window based on headBuffer

    tail : MPI.win
        window based on tailBuffer

    win : MPI.win
        MPI window based on buffer for stack

    """
    def __init__(self, comm, data, rank_mask=None, length=32, max_model_lag=None, failPush=False, name=None):
        """
        Parameters
        ----------
        comm : mpi4py.MPI.Comm
            Communicator for all ranks involved
        data : list
            Example data used to create buffer
        rank_mask : int, optional
            host of the window
        length : int, optional
            capacity of stack
        max_model_lag : int, optional
            Will not consider data past given model valide
        failPush : bool, optional
            Fail to overwrite data if queue is full
        name : string, optional
            name of constant for debbuging
        """
        self.MPI = ExaComm.get_MPI()
        self.comm = comm
        if rank_mask:
            self.rank = self.comm.rank
        self.length = length
        # This lets us fail a push when at full capacity
        # Otherwise will overwrite the oldest data
        self.failPush = failPush

        dataBytes = self.MPI.pickle.dumps(data)
        size = len(dataBytes) + 1
        super().__init__(bytes, size, comm_size=comm.size, max_model_lag=max_model_lag, name=name)

        self.buff = bytearray(self.dataSize)
        self.plus = np.array([1], dtype=np.int64)
        self.minus = np.array([-1], dtype=np.int64)

        totalSize = 0
        headSize = 0
        tailSize = 0

        # if comm.rank == rank:
        if rank_mask:
            totalSize = size * self.length
            headSize = self.MPI.INT64_T.Get_size()
            tailSize = self.MPI.INT64_T.Get_size()

        self.head = []
        self.tail = []
        self.win = []
        for i in range(comm.size):
            # Setup head window
            self.head.append(self.MPI.Win.Allocate(headSize, comm=self.comm.raw()))
            self.head[i].Lock(self.rank)
            self.head[i].Accumulate(
                np.zeros(1, dtype=np.int64), self.rank, op=self.MPI.REPLACE
            )
            self.head[i].Unlock(self.rank)
            self.head[i].Fence(self.rank)

            # Setup tail window
            self.tail.append(self.MPI.Win.Allocate(tailSize, comm=self.comm.raw()))
            self.tail[i].Lock(self.rank)
            self.tail[i].Accumulate(
                np.zeros(1, dtype=np.int64), self.rank, op=self.MPI.REPLACE
            )
            self.tail[i].Unlock(self.rank)
            self.tail[i].Fence(self.rank)

            # Setup data window
            self.win.append(
                self.MPI.Win.Allocate(totalSize, disp_unit=size, comm=self.comm.raw())
            )
            self.win[i].Fence(self.rank)

    def __del__(self):
        for i in range(self.comm.size):
            self.win[i].Free()
            self.head[i].Free()

    @introspectTrace(name=True)
    def pop(self, rank, count=1):
        """
        Returns data from head of stack if there is data.

        Parameters
        ----------
        rank : integer
            Host rank where to take data from

        count : integer
            How many pops to perform

        Returns
        -------
        list
            Data from stack if there is any.
        """
        ret = False
        head = np.zeros(1, dtype=np.int64)
        tail = np.zeros(1, dtype=np.int64)
        rank = int(rank)

        self.head[rank].Lock(self.rank)
        self.tail[rank].Lock(self.rank)

        # Read the head and tail pointers.
        reqHead = self.head[rank].Rget_accumulate(self.minus, head, self.rank, op=self.MPI.SUM)
        reqTail = self.tail[rank].Rget_accumulate(self.minus, tail, self.rank, op=self.MPI.NO_OP)
        reqHead.wait()
        reqTail.wait()
        # print("InPop", head[0], tail[0])
        if head[0] > tail[0]:
            ret = True
            index = (head[0] - 1) % self.length

            self.win[rank].Lock(self.rank)
            self.win[rank].Get_accumulate(
                self.buff,
                self.buff,
                self.rank,
                target=[index, self.dataSize],
                op=self.MPI.NO_OP,
            )
            self.win[rank].Unlock(self.rank)

        else:
            self.head[rank].Accumulate(
                self.plus, self.rank, op=self.MPI.SUM
            )

        self.tail[rank].Unlock(self.rank)
        self.head[rank].Unlock(self.rank)

        if ret:
            return self.MPI.pickle.loads(self.buff)
        return None

    @introspectTrace(name=True)
    def push(self, data, rank=None):
        """
        Pushes data to a rank's stack.

        Parameters
        ----------
        data : list
            Data to be pushed to rank's stack

        rank : integer, optional
            Rank to push data to

        Returns
        -------
        list
            Returns a capacity of stack and loss if data is overwritten
        """
        if rank is None:
            rank = self.comm.rank
        toSend = self.MPI.pickle.dumps(data)
        assert len(toSend) == self.dataSize

        head = np.zeros(1, dtype=np.int64)
        tail = np.zeros(1, dtype=np.int64)
        rank = int(rank)

        self.head[rank].Lock(self.rank)
        self.tail[rank].Lock(self.rank)

        # Read the head and tail pointers.
        reqHead = self.head[rank].Rget_accumulate(self.plus, head, self.rank, op=self.MPI.SUM)
        reqTail = self.tail[rank].Rget_accumulate(self.plus, tail, self.rank, op=self.MPI.NO_OP)
        reqHead.wait()
        reqTail.wait()

        # This is if we are going to loose data because we exceded capacity
        write = True
        if tail[0] + self.length == head[0]:
            if self.failPush:
                write = False
                self.head[rank].Accumulate(
                    self.minus, self.rank, op=self.MPI.SUM
                )
            else:
                self.tail[rank].Accumulate(
                    self.plus, self.rank, op=self.MPI.SUM
                )
            lost = 1
            capacity = self.length
        else:
            lost = 0
            capacity = head[0] - tail[0] + 1

        if write:
            # Actual write data
            index = head[0] % self.length
            self.win[rank].Lock(self.rank)
            self.win[rank].Accumulate(
                toSend, self.rank, target=[index, self.dataSize], op=self.MPI.REPLACE
            )
            self.win[rank].Unlock(self.rank)

        self.tail[rank].Unlock(self.rank)
        self.head[rank].Unlock(self.rank)
        return capacity, lost

class ExaMPICentralizedQueue(ExaData):
    """
    This class creates circular buffers in RMA windows across nodes in a communicator.
    There is a queue per rank.  Each rank acts as a host.

    Attributes
    ----------
    MPI : mpi4py.MPI
            mpi4py's MPI access point

    comm : mpi4py.MPI.Comm
        raw MPI communicator

    length : int
        capacity of the queue

    failPush : bool
        flag setting if push can overwrite data

    buff : bytearray
        internal buffer for queue used for RMA ops

    plus : np.array
        numpy constant for adding

    minus : np.array
        numpy constant for subtracting

    headBuffer : np.array
        buffer containing head counter

    tailBuffer : np.array
        buffer containing tail counter

    head : MPI.win
        window based on headBuffer

    tail : MPI.win
        window based on tailBuffer

    win : MPI.win
        MPI window based on buffer for queue

    """
    def __init__(self, comm, data, rank_mask=None, length=32, max_model_lag=None, failPush=False, name=None):
        """
        Parameters
        ----------
        comm : mpi4py.MPI.Comm
            Communicator for all ranks involved
        data : list
            Example data used to create buffer
        rank_mask : int, optional
            host of the window
        length : int, optional
            capacity of queue
        max_model_lag : int, optional
            Will not consider data past given model valide
        failPush : bool, optional
            Fail to overwrite data if queue is full
        name : string, optional
            name of constant for debbuging
        """
        self.MPI = ExaComm.get_MPI()
        self.comm = comm
        if rank_mask:
            self.rank = self.comm.rank
        self.length = length
        # This lets us fail a push when at full capacity
        # Otherwise will overwrite the oldest data
        self.failPush = failPush

        dataBytes = self.MPI.pickle.dumps(data)
        size = len(dataBytes) + 1
        super().__init__(bytes, size, comm_size=comm.size, max_model_lag=max_model_lag, name=name)

        self.buff = bytearray(self.dataSize)
        self.plus = np.array([1], dtype=np.int64)
        self.minus = np.array([-1], dtype=np.int64)

        totalSize = 0
        headSize = 0
        tailSize = 0
        # if comm.rank == rank:
        if rank_mask:
            totalSize = size * self.length
            headSize = self.MPI.INT64_T.Get_size()
            tailSize = self.MPI.INT64_T.Get_size()

        self.head = []
        self.tail = []
        self.win = []
        for i in range(comm.size):
            # Setup head window
            self.head.append(self.MPI.Win.Allocate(headSize, comm=self.comm.raw()))
            self.head[i].Lock(self.rank)
            self.head[i].Accumulate(
                np.zeros(1, dtype=np.int64), self.rank, op=self.MPI.REPLACE
            )
            self.head[i].Unlock(self.rank)
            self.head[i].Fence(self.rank)

            # Setup tail window
            self.tail.append(self.MPI.Win.Allocate(tailSize, comm=self.comm.raw()))
            self.tail[i].Lock(self.rank)
            self.tail[i].Accumulate(
                np.zeros(1, dtype=np.int64), self.rank, op=self.MPI.REPLACE
            )
            self.tail[i].Unlock(self.rank)
            self.tail[i].Fence(self.rank)

            # Setup data window
            self.win.append(
                self.MPI.Win.Allocate(totalSize, disp_unit=size, comm=self.comm.raw())
            )
            self.win[i].Fence(self.rank)

    def __del__(self):
        for i in range(self.comm.size):
            self.win[i].Free()
            self.head[i].Free()

    @introspectTrace(name=True)
    def pop(self, rank, count=1):
        """
        Returns data from head of queue if there is data.

        Parameters
        ----------
        rank : integer
            Host rank where to take data from

        count : integer, optional
            How many pops to perform

        Returns
        -------
        list
            Data from queue if there is any.
        """
        ret = True
        head = np.zeros(1, dtype=np.int64)
        tail = np.zeros(1, dtype=np.int64)
        rank = int(rank)

        self.head[rank].Lock(self.rank)
        self.tail[rank].Lock(self.rank)

        # Read the head and tail pointers.
        reqHead = self.head[rank].Rget_accumulate(self.minus, head, self.rank, op=self.MPI.NO_OP)
        reqTail = self.tail[rank].Rget_accumulate(self.plus, tail, self.rank, op=self.MPI.SUM)
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
                op=self.MPI.NO_OP,
            )
            self.win[rank].Unlock(self.rank)
        else:
            # Dec the tail pointer
            self.tail[rank].Accumulate(self.minus, self.rank, op=self.MPI.SUM)
            ret = False

        self.tail[rank].Unlock(self.rank)
        self.head[rank].Unlock(self.rank)

        if ret:
            return self.MPI.pickle.loads(self.buff)
        return None

    @introspectTrace(name=True)
    def push(self, data, rank=None):
        """
        Pushes data to a rank's queue.

        Parameters
        ----------
        data : list
            Data to be pushed to rank's queue

        rank : integer, optional
            Rank to push data to

        Returns
        -------
        list
            Returns a capacity of queue and loss if data is overwritten
        """
        if rank is None:
            rank = self.comm.rank
        toSend = self.MPI.pickle.dumps(data)
        assert len(toSend) <= self.dataSize

        head = np.zeros(1, dtype=np.int64)
        tail = np.zeros(1, dtype=np.int64)

        self.head[rank].Lock(self.rank)
        self.tail[rank].Lock(self.rank)

        reqHead = self.head[rank].Rget_accumulate(self.plus, head, self.rank, op=self.MPI.SUM)
        reqTail = self.tail[rank].Rget_accumulate(self.plus, tail, self.rank, op=self.MPI.NO_OP)
        reqHead.wait()
        reqTail.wait()

        write = True
        headIndex = head[0] % self.length
        tailIndex = tail[0] % self.length
        if head[0] > tail[0] and headIndex == tailIndex:
            if self.failPush:
                write = False
                self.head[rank].Accumulate(
                    self.minus, self.rank, op=self.MPI.SUM
                )
            else:
                self.tail[rank].Accumulate(
                    self.plus, self.rank, op=self.MPI.SUM
                )
            lost = 1
            capacity = self.length
        else:
            lost = 0
            capacity = head[0] - tail[0]

        if write:
            self.win[rank].Lock(self.rank)
            self.win[rank].Accumulate(
                toSend, self.rank, target=[headIndex, len(toSend)], op=self.MPI.REPLACE
            )
            self.win[rank].Unlock(self.rank)

        self.tail[rank].Unlock(self.rank)
        self.head[rank].Unlock(self.rank)

        return capacity, lost
