from exarl.utils.introspect import ib
from exarl.utils.introspect import introspectTrace
from exarl.base.comm_base import ExaComm
import os
import numpy as np

import exarl.utils.candleDriver as cd
workflow = cd.run_params['workflow']
if workflow == 'async':
    import mpi4py.rc
    mpi4py.rc.threads = False
    mpi4py.rc.recv_mprobe = False
from mpi4py import MPI

class ExaSimple(ExaComm):
    MPI = MPI

    def __init__(self, comm=MPI.COMM_WORLD, procs_per_env=1, num_learners=1):
        if comm is None:
            self.comm = MPI.COMM_WORLD
            self.size = MPI.COMM_WORLD.Get_size()
            self.rank = MPI.COMM_WORLD.Get_rank()
        else:
            self.comm = comm
            self.size = comm.size
            self.rank = comm.rank

        # if self.rank > 0:
        #     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.buffers = {}
        super().__init__(self, procs_per_env, num_learners)

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

    def split(self, procs_per_env, num_learners):
        # Agent communicator
        agent_color = MPI.UNDEFINED
        if (self.rank < num_learners) or ((self.rank + procs_per_env - 1) % procs_per_env == 0):
            agent_color = 0
        agent_comm = self.comm.Split(agent_color, self.rank)
        if agent_color == 0:
            agent_comm = ExaSimple(comm=agent_comm)
        else:
            agent_comm = None

        # Environment communicator
        if self.rank < num_learners:
            env_color = 0
        else:
            env_color = (int((self.rank - num_learners) / procs_per_env)) + 1
        env_comm = ExaSimple(comm=self.comm.Split(env_color, self.rank))

        # Learner communicator
        learner_color = MPI.UNDEFINED
        if self.rank < num_learners:
            learner_color = 0
        learner_comm = self.comm.Split(learner_color, self.rank)
        if learner_color == 0:
            learner_comm = ExaSimple(comm=learner_comm)
        else:
            learner_comm = None

        return agent_comm, env_comm, learner_comm

    def raw(self):
        return self.comm
