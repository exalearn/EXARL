import sys
from importlib import reload
from exarl.utils.globals import ExaGlobals
from exarl.base.comm_base import ExaComm
from exarl.utils.introspect import introspectTrace

class ExaSimple(ExaComm):
    """
    This class is built as a simple wrapper around mpi4py.
    Instances are a type of ExaComm which is used to send,
    receive, and synchronize data across the participating
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

    MPI = None

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

        # Singleton
        if ExaSimple.MPI is None:
            print("DOING THE IMPORT!!!!!!!!!!")
            mpi4py_rc = True if ExaGlobals.lookup_params('mpi4py_rc') in ["true", "True", 1] else False
            if mpi4py_rc:
                print("Turning mpi4py.rc.threads and mpi4py.rc.recv_mprobe to false!", flush=True)
                import mpi4py.rc
                mpi4py.rc.threads = False
                mpi4py.rc.recv_mprobe = False
            if "mpi4py" not in sys.modules:
                import mpi4py
            else:
                mpi4py = reload(sys.modules["mpi4py"])
            from mpi4py import MPI
            ExaSimple.MPI = MPI

        if comm is None:
            self.comm = ExaSimple.MPI.COMM_WORLD
            self.size = ExaSimple.MPI.COMM_WORLD.Get_size()
            self.rank = ExaSimple.MPI.COMM_WORLD.Get_rank()
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
        return self.comm.send(data, dest=dest)

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
            source = ExaSimple.MPI.ANY_SOURCE
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
        converter = {sum: ExaSimple.MPI.SUM, 
                     max: ExaSimple.MPI.MAX, 
                     min: ExaSimple.MPI.MIN}
        return self.comm.reduce(arg, op=converter[op], root=root)

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
        converter = {sum: ExaSimple.MPI.SUM, 
                     max: ExaSimple.MPI.MAX, 
                     min: ExaSimple.MPI.MIN}
        if op is None:
            op = ExaSimple.MPI.LAND
        elif op in converter:
            op = converter[sum]
        return self.comm.allreduce(arg, op)

    def time(self):
        """
        Returns MPI wall clock time
        """
        return ExaSimple.MPI.Wtime()

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

        if ExaSimple.MPI.COMM_WORLD.Get_size() == procs_per_env:
            assert num_learners == 1, "num_learners should be 1 when global comm size == procs_per_env"
            color = ExaSimple.MPI.UNDEFINED
            if self.rank == 0:
                color = 0
            learner_comm = self.comm.Split(color, self.rank)
            agent_comm = self.comm.Split(color, self.rank)
            if self.rank == 0:
                learner_comm = ExaSimple(learner_comm, procs_per_env, num_learners)
                agent_comm = ExaSimple(agent_comm, procs_per_env, num_learners)
            else:
                learner_comm = None
                agent_comm = None

            env_color = 0
            env_comm = self.comm.Split(env_color, self.rank)
            env_comm = ExaSimple(env_comm, procs_per_env, num_learners)
        else:
            # Agent communicator
            agent_color = ExaSimple.MPI.UNDEFINED
            if (self.rank < num_learners) or ((self.rank - num_learners) % procs_per_env == 0):
                agent_color = 0
            agent_comm = self.comm.Split(agent_color, self.rank)
            if agent_color == 0:
                agent_comm = ExaSimple(agent_comm, procs_per_env, num_learners)
            else:
                agent_comm = None

            # Environment communicator
            if self.rank < num_learners:
                env_color = 0
            else:
                env_color = (int((self.rank - num_learners) / procs_per_env)) + 1
            env_comm = self.comm.Split(env_color, self.rank)
            if env_color > 0:
                env_comm = ExaSimple(env_comm, procs_per_env, num_learners)
            else:
                env_comm = None

            # Learner communicator
            learner_color = ExaSimple.MPI.UNDEFINED
            if self.rank < num_learners:
                learner_color = 0
            learner_comm = self.comm.Split(learner_color, self.rank)
            if learner_color == 0:
                learner_comm = ExaSimple(learner_comm, procs_per_env, num_learners)
            else:
                learner_comm = None

        return agent_comm, env_comm, learner_comm

    def raw(self):
        """
        Returns raw MPI comm
        """
        return self.comm
