from exarl.utils.introspect import ib
from exarl.utils.introspect import introspectTrace
from exarl.base.comm_base import ExaComm
import os
import numpy as np

import exarl.candle.candleDriver as cd
workflow = cd.run_params['workflow']
if workflow == 'async':
    print("Turning mpi4py.rc.threads and mpi4py.rc.recv_mprobe to false!")
    import mpi4py.rc
    mpi4py.rc.threads = False
    mpi4py.rc.recv_mprobe = False
from mpi4py import MPI

class ExaSimple(ExaComm):
    """
    This class is built as a simple wrapper around mpi4py.
    Instances are a type of ExaComm which is used to send,
    recieve, and synchronize data across the participating
    ranks.

    Attributes
    ----------
    MPI : MPI
        MPI class used to access comm, sizes, and rank

    comm : MPI.comm
        The underlying communicator

    size : int
        Number of processes in the communicator

    rank : int
        Rank of the current process

    """

    MPI = MPI

    def __init__(self, comm=MPI.COMM_WORLD, procs_per_env=1, num_learners=1):
        """
        Parameters
        ----------
        comm : MPI Comm, optional
            The base MPI comm to split into sub-comms.  If set to None
            will default to MPI.COMM_WORLD
        procs_per_env : int, optional
            Number of processes per environment (sub-comm)
        num_learners : int, optional
            Number of learners (multi-learner)
        """

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
        """
        Point-to-point communication between ranks. Send must have
        matching recv.

        Parameters
        ----------
        data : any
            Data to be sent
        dest : int
            Rank within comm where data will be sent.
        pack : int, optional
            Not used
        """
        return self.comm.send(data, dest=dest)

    @introspectTrace()
    def recv(self, data, source=MPI.ANY_SOURCE):
        """
        Point-to-point communication between ranks. Recv must have
        matching send.

        Parameters
        ----------
        data : any
            Not used
        dest : int
            Rank within comm where data will be sent. Must have matching recv.
        source : int, optional
            Rank to recieve data from.  Default allows data from any source.
        """
        return self.comm.recv(source=source)

    @introspectTrace()
    def bcast(self, data, root):
        """
        Broadcasts data from the root to all other processes in comm.

        Parameters
        ----------
        data : any
            Data to be broadcast
        root : int
            Indicate which process data comes from
        """
        return self.comm.bcast(data, root=root)

    def barrier(self):
        """
        Block synchronization for the comm.
        """
        return self.comm.Barrier()

    def reduce(self, arg, op, root):
        """
        Data is joined from all processes in comm by doing op.
        Result is placed on root.

        Parameters
        ----------
        arg : any
            Data to reduce
        op : str
            Supports sum, max, and min reductions
        root : int
            Rank the result will end on
        """
        converter = {sum: MPI.SUM, max: MPI.MAX, min: MPI.MIN}
        return self.comm.reduce(arg, op=converter[op], root=root)

    def allreduce(self, arg, op=MPI.LAND):
        """
        Data is joined from all processes in comm by doing op.
        Data is put on all processes in comm.

        Parameters
        ----------
        arg : any
            Data to reduce
        op : MPI op, optional
            Operation to perform
        """
        return self.comm.allreduce(arg, op)

    def time(self):
        """
        Returns MPI wall clock time
        """
        return MPI.Wtime()

    def split(self, procs_per_env, num_learners):
        """
        This splits the comm into agent, environment, and learner comms.
        Returns three simple sub-comms

        Parameters
        ----------
        procs_per_env : int
            Number of processes per environment comm
        num_learners : int
            Number of processes per learner comm
        """
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
        """
        Returns raw MPI comm
        """
        return self.comm
