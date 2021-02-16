from utils.introspect import ib
from utils.introspect import introspectTrace
import numpy as np
import exarl as erl

import mpi4py.rc

mpi4py.rc.threads = False
mpi4py.rc.recv_mprobe = False
from mpi4py import MPI


class ExaSimple(erl.ExaComm):
    def __init__(self, comm=MPI.COMM_WORLD, procs_per_env=1):
        if comm is None:
            comm = MPI.COMM_WORLD
        self.comm = comm
        self.size = comm.Get_size()
        self.rank = comm.Get_rank()
        self.buffers = {}
        super().__init__(self, procs_per_env)

    def list_like(self, data, prep=True):
        if isinstance(data, range):
            list_flag = True
            cast_flag = True
        elif isinstance(data, tuple):
            list_flag = True
            cast_flag = True
        elif isinstance(data, np.ndarray):
            list_flag = True
            cast_flag = False
        elif isinstance(data, list):
            list_flag = True
            cast_flag = False
        else:
            list_flag = False
            cast_flag = False

        if prep:
            toSend = data
            if cast_flag:
                toSend = list(data)
            elif not list_flag:
                toSend = [data]
            return list_flag, toSend
        return list_flag

    def is_float(self, data):
        list_flag, new_data = self.list_like(data)
        if not list_flag:
            if (
                isinstance(data, float)
                or isinstance(data, np.float32)
                or isinstance(data, np.float64)
            ):
                return True
            return False
        return any([self.is_float(x) for x in data])

    def get_flat_size(self, data):
        list_flag, new_data = self.list_like(data)
        if not list_flag:
            return 1
        return sum([self.get_flat_size(x) for x in data])

    def encode_list_format(self, data, buff=None, np_arrays=None, level=0):
        if not self.list_like(data, prep=False):
            return self.encode_type(type(data)), np_arrays

        # Everything after this should be list like!
        if buff is None:
            buff = []
        if np_arrays is None:
            np_arrays = []

        # 0 indicates a new list
        # 2 indecates a new np.array
        # 1 end of list (either list or np.array)
        if isinstance(data, np.ndarray):
            np_arrays.append(np.shape(data))
            data = data.flatten()
            # buff.append(2)
        # else:
        # buff.append(0)
        for i, x in enumerate(data):
            if self.list_like(data[i], prep=False):
                self.encode_list_format(
                    data[i], buff=buff, np_arrays=np_arrays, level=level + 1
                )
            # else:
            #     buff.append(self.encode_type(type(x)))
        # buff.append(1)
        return buff, np_arrays

    @introspectTrace()
    def send(self, data, dest, pack=False):
        return self.comm.send(data, dest=dest)

    @introspectTrace()
    def recv(self, data, source=MPI.ANY_SOURCE):
        return self.comm.recv(source=source)

    @introspectTrace()
    def bcast(self, data, root):
        return self.comm.bcast(data, root=root)

    def barrier(self):
        return self.comm.Barrier()

    def reduce(self, arg, op, root):
        converter = {sum: MPI.SUM, max: MPI.MAX, min: MPI.MIN}
        return self.comm.reduce(arg, op=converter[op], root=root)

    def allreduce(self, arg, op=MPI.LAND):
        return self.comm.allreduce(arg, op)

    def gather(self, data, root):
        return self.comm.gather(data, root=root)

    def time(self):
        return MPI.Wtime()

    def split(self, procs_per_env):
        # Agent communicator
        agent_color = MPI.UNDEFINED
        if (self.rank == 0) or ((self.rank + procs_per_env - 1) % procs_per_env == 0):
            agent_color = 0
        agent_comm = self.comm.Split(agent_color, self.rank)

        # Environment communicator
        if self.rank == 0:
            env_color = 0
        else:
            env_color = (int((self.rank - 1) / procs_per_env)) + 1
        env_comm = self.comm.Split(env_color, self.rank)

        if agent_color == 0:
            agent_comm = ExaSimple(comm=agent_comm)
        else:
            agent_comm = None
        env_comm = ExaSimple(comm=env_comm)
        return agent_comm, env_comm

    def raw(self):
        return self.comm


# class LazyCallable(object):
#   def __init__(self, name):
#     self.n, self.f = name, None

#   def __call__(self, *a, **k):
#     if self.f is None:
#       modn, funcn = self.n.rsplit('.', 1)
#       if modn not in sys.modules:
#         __import__(modn)
#       self.f = getattr(sys.modules[modn],
#                        funcn)
#     self.f(*a, **k)
