import sys
import pickle
from exarl.utils.globals import ExaGlobals
from exarl.base.comm_base import ExaComm
from exarl.utils.introspect import introspectTrace
import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI

class ExaRsocket(ExaComm):
    """
    This class is ...

    Attributes
    ----------
    size : int
        Number of processes in the communicator

    rank : int
        Rank of the current process

    """

    MPI = MPI

    def __init__(self, comm, procs_per_env, num_learners):
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
        import pysocketserver
        if comm is None:
            print("EXARSOCKET: about to create", flush=True)
            self.comm = pysocketserver.Communicator()
            print("EXARSOCKET: Done creating", flush=True)
            self.size = self.comm.size()
            self.rank = self.comm.rank()
        else:
            self.comm = comm
            self.size = comm.size
            self.rank = comm.rank

        super().__init__(self, procs_per_env, num_learners)

    @introspectTrace()
    def send(self, data, dest, pack=False):
        """
        Point-to-point communication between ranks.  Send must have
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
        toSend = pickle.dumps(data)
        return self.comm.send(toSend, len(toSend), dest, True)

    @introspectTrace()
    def recv(self, data, source=None):
        """
        Point-to-point communication between ranks.  Send must have
        matching send.

        Parameters
        ----------
        data : any
            Not use
        source : int, optional
            Rank to receive data from.  Default allows data from any source.
        """
        if source is None:
            toRecv = self.comm.recv()
        else:
            toRecv = self.comm.recv(source)
        return pickle.loads(toRecv)

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
        if root == self.rank:
            toSend = pickle.dumps(data)
            for i in range(self.comm.size):
                if i != self.comm.rank:
                    self.comm.send(toSend, len(toSend), i, True)
        else:
            toRecv = self.comm.recv(root)
            return pickle.loads(toRecv)
        return data

    def barrier(self):
        """
        Block synchronization for the comm.
        """
        return self.comm.barrier()

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
        pass
        # converter = {sum: ExaSimple.MPI.SUM,
        #              max: ExaSimple.MPI.MAX,
        #              min: ExaSimple.MPI.MIN}
        # return self.comm.reduce(arg, op=converter[op], root=root)

    def allreduce(self, arg, op=None):
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
        # converter = {sum: ExaSimple.MPI.SUM,
        #              max: ExaSimple.MPI.MAX,
        #              min: ExaSimple.MPI.MIN}
        # if op is None:
        #     op = ExaSimple.MPI.LAND
        # elif op in converter:
        #     op = converter[sum]
        # return self.comm.allreduce(arg, op)
        pass

    def time(self):
        """
        Returns MPI wall clock time
        """
        return self.comm.time()

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
        return None, None, None
        # if self.comm.size == procs_per_env:
        #     assert num_learners == 1, "num_learners should be 1 when global comm size == procs_per_env"
        #     env_comm = pysocketserver.Communicator(self.comm.maxPort() + 1)

        #     if self.rank == 0:
        #         learner_comm = pysocketserver.Communicator(env_comm.maxPort() + 1)
        #         agent_comm = pysocketserver.Communicator(learner_comm.maxPort() + 1)
        #     else:
        #         learner_comm = None
        #         agent_comm = None
        # else:
        #     maxPort = self.comm.maxPort()
            
        #     # Agent communicator
        #     agent_flag = (self.rank < num_learners) or ((self.rank - num_learners) % procs_per_env == 0)
        #     agent_comm = pysocketserver.Communicator.split(self.comm, agent_flag, maxPort)
        #     if agent_flag:
        #         maxPort = agent_comm.maxPort()
        #     else:
        #         agent_comm = None

        #     # Environment communicator
        #     if self.rank < num_learners:
        #         env_color = -1
        #     else:
        #         env_color = (int((self.rank - num_learners) / procs_per_env))

        #     env_comm = None
        #     numEnvSplits = (int(((self.size - 1) - num_learners) / procs_per_env)) + 1
        #     for i in range(0, numEnvSplits):
        #         env_flag = (env_color == i)
        #         temp_comm = pysocketserver.Communicator.split(self.comm, env_flag, maxPort)
        #         if env_flag:
        #             env_comm = temp_comm
        #             maxPort = temp_comm.maxPort()

        #     # Learner communicator
        #     learner_flag = self.rank < num_learners
        #     learner_comm = pysocketserver.Communicator.split(self.comm, learner_flag, maxPort)
        #     if not learner_flag:
        #         learner_comm = None

        # return agent_comm, env_comm, learner_comm

    def raw(self):
        """
        Returns raw MPI comm
        """
        return self.comm

    def shutdown(self):
        del self.comm