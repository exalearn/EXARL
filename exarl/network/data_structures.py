from mpi4py import MPI
import numpy as np
class MPI_RMA_QUEUE():
    def __init__(self, comm, is_learner, size=None, data=None, length=32, failPush=True, usePopAll=False):
        self.comm = comm
        self.length = length
        # This lets us fail a push when at full capacity
        # Otherwise will overwrite the oldest data
        self.failPush = failPush

        if data is not None:
            dataBytes = MPI.pickle.dumps(data)
            size = len(dataBytes)

        self.dataSize = size

        # Setup buffers
        self.buff = bytearray(self.dataSize)
        self.big_buff = bytearray(self.dataSize * self.length) if usePopAll else None
        self.second_big_buff = bytearray(self.dataSize * self.length) if usePopAll else None

        # Setup variables
        self.req_buff = None
        self.req_big_buff = None
        self.req_second_big_buff = None
        self.req_tail = None
        self.size_big_buff = 0
        self.size_second_big_buff = 0

        # Setup constants
        self.plus = np.array([1], dtype=np.int64)
        self.minus = np.array([-1], dtype=np.int64)

        # Setup RMA windows
        self.headBuff = None
        self.tailBuff = None
        totalSize = 0
        disp = MPI.DOUBLE.Get_size()
        index_win_size = 0
        disp_win_index = MPI.LONG.Get_size()
        if not is_learner:
            totalSize = size * self.length
            self.headBuff = np.zeros(1, dtype=np.int64)
            self.tailBuff = np.zeros(1, dtype=np.int64)

            index_win_size = disp_win_index

        # Setup head window
        self.head = MPI.Win.Allocate(index_win_size, disp_win_index, comm=self.comm)

        # Setup tail window
        self.tail = MPI.Win.Allocate(index_win_size, disp_win_index, comm=self.comm)

        # Setup data window
        self.win = MPI.Win.Allocate(totalSize, disp_unit=size, comm=self.comm)

        # Initialize RMA windows
        if not is_learner:
            self.head.Lock(self.comm.rank)
            self.head.Put(self.headBuff, target_rank=self.comm.rank)
            self.head.Unlock(self.comm.rank)

            self.tail.Lock(self.comm.rank)
            self.tail.Put(self.tailBuff, target_rank=self.comm.rank)
            self.tail.Unlock(self.comm.rank)


    def __del__(self):
        self.win.Free()
        self.tail.Free()
        self.head.Free()

    def push(self, data, rank=None):
        if rank is None:
            rank = self.comm.rank
        toSend = MPI.pickle.dumps(data)
        assert len(toSend) <= self.dataSize

        head = np.zeros(1, dtype=np.int64)
        tail = np.zeros(1, dtype=np.int64)


        self.head.Lock(rank)
        self.tail.Lock(rank)
        self.head.Get_accumulate(self.plus, head, rank, op=MPI.SUM)
        self.tail.Get_accumulate(self.plus, tail, rank, op=MPI.NO_OP)

        self.head.Flush(rank)
        self.tail.Flush(rank)

        write = True
        headIndex = head[0] % self.length
        tailIndex = tail[0] % self.length
        if head[0] > tail[0] and headIndex == tailIndex:
            if self.failPush:
                write = False
                self.head.Accumulate(
                    self.minus, rank, op=MPI.SUM
                )
            else:
                self.tail.Accumulate(
                    self.plus, rank, op=MPI.SUM
                )
            lost = 1
            capacity = self.length
        else:
            lost = 0
            capacity = head[0] - tail[0]

        if write:
            self.win.Lock(rank)
            self.win.Accumulate(
                toSend, rank, target=[headIndex, len(toSend)], op=MPI.REPLACE
            )
            self.win.Unlock(rank)

        self.tail.Unlock(rank)
        self.head.Unlock(rank)

        return capacity, lost

    def pop(self, rank, count=1):
        ret = True
        head = np.zeros(1, dtype=np.int64)
        tail = np.zeros(1, dtype=np.int64)
        rank = int(rank)

        self.head.Lock(rank)
        self.tail.Lock(rank)

        # Read the head and tail pointers.
        self.head.Get_accumulate(self.minus, head, rank, op=MPI.NO_OP)
        self.tail.Get_accumulate(self.plus, tail, rank, op=MPI.SUM)

        self.head.Flush(rank)
        self.tail.Flush(rank)

        # Is there data
        if head[0] > tail[0]:
            index = tail[0] % self.length
            self.win.Lock(rank)
            self.win.Get_accumulate(
                self.buff,
                self.buff,
                rank,
                target=[index, self.dataSize],
                op=MPI.NO_OP,
            )
            self.win.Unlock(rank)
        else:
            # Dec the tail pointer
            self.tail.Accumulate(self.minus, rank, op=MPI.SUM)
            ret = False

        self.tail.Unlock(rank)
        self.head.Unlock(rank)

        if ret:
            return True, MPI.pickle.loads(self.buff)

        return False, None

    def pop_all(self, rank):
        ret = True
        head = np.zeros(1, dtype=np.int64)
        tail = np.zeros(1, dtype=np.int64)
        rank = int(rank)

        self.head.Lock(rank)
        self.tail.Lock(rank)
        # Read the head and tail pointers.
        self.head.Get_accumulate(self.minus, head, rank, op=MPI.NO_OP)
        self.tail.Get_accumulate(self.plus, tail, rank, op=MPI.NO_OP)
        self.head.Flush(rank)
        self.tail.Flush(rank)

        elem_nb = head[0] - tail[0]
        # Is there data
        grouped_data = []
        if elem_nb > 0:
            reqTail = self.tail.Raccumulate( np.array([elem_nb], dtype=np.int64), rank, op=MPI.SUM)
            index = tail[0] % self.length
            self.win.Lock(rank)
            if index + elem_nb <= self.length : # liniar reading
                self.win.Get_accumulate(
                    self.buff,
                    self.big_buff,
                    rank,
                    target=[index, elem_nb * self.dataSize],
                    op=MPI.NO_OP,
                )
                self.win.Unlock(rank)
                grouped_data = []
                for i in range(elem_nb):
                    grouped_data.append(MPI.pickle.loads(self.big_buff[i*self.dataSize:(i+1)*self.dataSize]) )

            else: # circular reading
                first_part_length = (self.length - index)*self.dataSize
                self.win.Get_accumulate(
                    self.buff,
                    self.big_buff,
                    rank,
                    target=[index, first_part_length],
                    op=MPI.NO_OP,
                )
                self.win.Flush(rank)
                for i in range(self.length - index):
                    grouped_data.append(MPI.pickle.loads(self.big_buff[i*self.dataSize:(i+1)*self.dataSize]) )

                second_part_length = (head[0] % self.length) * self.dataSize
                self.win.Get_accumulate(
                    self.buff,
                    self.second_big_buff,
                    rank,
                    target=[0, second_part_length],
                    op=MPI.NO_OP,
                )
                self.win.Unlock(rank)
                for i in range(head[0] % self.length):
                    grouped_data.append(MPI.pickle.loads(self.second_big_buff[i*self.dataSize:(i+1)*self.dataSize]))

            reqTail.wait()

        else:
            ret = False

        self.tail.Unlock(rank)
        self.head.Unlock(rank)

        if ret:
            return True, grouped_data
        else:
            return False, None

    def request_pop(self, rank, count=1):
        ret = True
        head = np.zeros(1, dtype=np.int64)
        tail = np.zeros(1, dtype=np.int64)
        rank = int(rank)

        self.head.Lock(rank)
        self.tail.Lock(rank)

        # Read the head and tail pointers.

        self.head.Get_accumulate(self.minus, head, rank, op=MPI.NO_OP)
        self.tail.Get_accumulate(self.plus, tail, rank, op=MPI.SUM)
        self.head.Flush(rank)
        self.tail.Flush(rank)

        # Is there space
        if head[0] > tail[0]:
            index = tail[0] % self.length
            self.win.Lock(rank)
            self.req_buff = self.win.Rget_accumulate(
                self.buff,
                self.buff,
                rank,
                target=[index, self.dataSize],
                op=MPI.NO_OP,
            )

        else:
            # Dec the tail pointer
            self.tail.Accumulate(self.minus, rank, op=MPI.SUM)

            # unlock windos
            self.tail.Unlock(rank)
            self.head.Unlock(rank)

            self.req_buff = None
            ret = False
        return ret

    def wait_pop(self, rank):

        self.win.Unlock(rank)
        self.tail.Unlock(rank)
        self.head.Unlock(rank)

        return MPI.pickle.loads(self.buff)

    def request_pop_all(self, rank):
        ret = True
        head = np.zeros(1, dtype=np.int64)
        tail = np.zeros(1, dtype=np.int64)
        rank = int(rank)

        self.head.Lock(rank)
        self.tail.Lock(rank)
        # Read the head and tail pointers.
        self.head.Get_accumulate(self.minus, head, rank, op=MPI.NO_OP)
        self.tail.Get_accumulate(self.plus, tail, rank, op=MPI.NO_OP)
        self.head.Flush(rank)
        self.tail.Flush(rank)

        elem_nb = head[0] - tail[0]

        # Is there data
        if elem_nb > 0:
            self.req_tail = self.tail.Raccumulate( np.array([elem_nb], dtype=np.int64), rank, op=MPI.SUM)
            index = tail[0] % self.length
            self.win.Lock(rank)
            if index + elem_nb <= self.length : # liniar reading
                self.req_big_buff = self.win.Rget_accumulate(
                    self.buff,
                    self.big_buff,
                    rank,
                    target=[index, elem_nb * self.dataSize],
                    op=MPI.NO_OP,
                )
                self.req_second_big_buff = None

                self.size_big_buff = elem_nb
                self.size_second_big_buff = 0

            else: # circular reading
                # First buffer part
                first_part_length = (self.length - index)*self.dataSize
                self.req_big_buff = self.win.Rget_accumulate(
                    self.buff,
                    self.big_buff,
                    rank,
                    target=[index, first_part_length],
                    op=MPI.NO_OP,
                )
                # Second buffer part
                second_part_length = (head[0] % self.length) * self.dataSize
                self.req_second_big_buff = self.win.Rget_accumulate(
                    self.buff,
                    self.second_big_buff,
                    rank,
                    target=[0, second_part_length],
                    op=MPI.NO_OP,
                )

                self.size_big_buff = self.length - index
                self.size_second_big_buff = head[0] % self.length

        else:
            ret = False
            self.tail.Unlock(rank)
            self.head.Unlock(rank)
            self.req_tail = None
            self.req_big_buff = None
            self.req_second_big_buff = None

        return ret

    def wait_pop_all(self, rank):

        if self.req_tail is not None :
            self.req_tail.wait()

        if self.req_big_buff is not None :
            self.req_big_buff.wait()

        if self.req_second_big_buff is not None :
            self.req_second_big_buff.wait()

        self.win.Unlock(rank)
        self.tail.Unlock(rank)
        self.head.Unlock(rank)

        grouped_data = []
        if self.req_big_buff is not None :

            if self.req_second_big_buff is not None:
                for i in range(self.size_big_buff):
                    grouped_data.append(MPI.pickle.loads(self.big_buff[i*self.dataSize:(i+1)*self.dataSize]) )

                for i in range(self.size_second_big_buff):
                    grouped_data.append(MPI.pickle.loads(self.second_big_buff[i*self.dataSize:(i+1)*self.dataSize]))
            else:
                grouped_data = [MPI.pickle.loads(self.big_buff[i*self.dataSize:(i+1)*self.dataSize]) for i in range(self.size_big_buff)]



        return grouped_data
