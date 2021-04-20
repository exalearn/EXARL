from utils.introspect import ib
from utils.introspect import introspectTrace
from utils.typing import TypeUtils
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
