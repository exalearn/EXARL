import sys
from exarl.utils.globals import ExaGlobals
from exarl.base.comm_base import ExaComm
from exarl.network.affinity import Affinity
from exarl.utils.introspect import introspectTrace
import mpi4py

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
            mpi4py_rc = True if ExaGlobals.lookup_params('mpi4py_rc') in ["true", "True", 1] else False
            if not mpi4py_rc:
                print("Turning mpi4py.rc.threads and mpi4py.rc.recv_mprobe to false!", flush=True)
                mpi4py.rc.threads = False
                mpi4py.rc.recv_mprobe = False
            # This statement actually starts MPI assuming this is the first call
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

    def split(self, procs_per_env, num_learners, affinity=None):
        affinity = Affinity(ExaSimple.MPI.COMM_WORLD, self.procs_per_env, self.num_learners)
        return super().split(procs_per_env, num_learners, affinity=affinity)

    def raw(self):
        """
        Returns raw MPI comm
        """
        return self.comm
