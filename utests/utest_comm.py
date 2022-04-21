import pytest
from exarl.utils.globals import ExaGlobals
from exarl.utils.candleDriver import initialize_parameters
from exarl.base.comm_base import ExaComm
from exarl.network.simple_comm import ExaSimple
from exarl.network.mpi_comm import ExaMPI

import mpi4py
# Unfortunatly, this line is starting MPI instead of the communicators.
# I can't figure out how to parameterize a fixture from a fixture which
# ultimately causes the problem.
from mpi4py import MPI

class TestCommHelper:
    """"
    This is a helper class with constants used throughout the comm tests.

    Attributes
    ----------
    comm_types : list
        List of comm types to test
    """
    comm_types = [ExaSimple, ExaMPI]

    def get_configs():
        """
        This is a generator that spits out configurations of learners, agents, and procs per agent.
        This is used to generate tests for split.  If there are no configurations (i.e. size=1)
        then nothing will be returned and the test will be skipped

        Returns
        -------
        Pair
            Number of learners and proccesses per environment for comm setup
        """
        size = MPI.COMM_WORLD.Get_size()
        yield 1, size
        for num_learners in range(1, size):
            rem = size - num_learners
            # Iterate over all potential procs_per_env counts
            for i in range(0, rem):
                # Add one since we want the size of the env_count not index
                procs_per_env = i + 1
                # Does it fit, then return it
                if rem % procs_per_env == 0:
                    yield num_learners, procs_per_env

@pytest.fixture(scope="session")
def mpi4py_rc(pytestconfig):
    """
    This function sets up the mpi4py import.

    Attributes
    ----------
    pytestconfig :
        Parameters passed from pytest.
    """
    mpi_flag = pytestconfig.getoption("mpi4py_rc")
    initialize_parameters(params={"mpi4py_rc": mpi_flag,
                                  "log_level": [3, 3]})
    return mpi_flag

@pytest.fixture(scope="function", autouse=True)
def reset_comm(mpi4py_rc):
    """
    This decorator is used to reset the comm before each function
    """
    ExaComm.reset()
    assert ExaComm.global_comm is None
    assert ExaComm.agent_comm is None
    assert ExaComm.env_comm is None
    assert ExaComm.num_learners == 1
    assert ExaComm.procs_per_env == 1

class TestEnvMembers:
    """
    This class test the basic functionality of the ExaComm class.
    These tests focus on testing the make up of a comm as well
    as its initialization.
    We are omitting test for the message passing and synchronization
    for now.
    TODO: Write tests for general functionality!
    """

    @pytest.mark.parametrize("comm", TestCommHelper.comm_types)
    def test_MPI_flags(self, comm, mpi4py_rc):
        """
        THIS TEST IS TO CHECK THAT THE MPI LAYERS SET THE FOLLOWING FLAGS!!!
        In reality this test will look to see the first comm imported and
        if it sets these flags.  Subsequent imports will not help.  I am
        unsure how to unload and reload modules in meaningful way such that
        it resets these flags.  This is why I have the simple_comm loaded
        before mpi_comm or mpi4py since it is the comm that is used 99% of
        the time.  If this test fails, PUT THE FLAGS BACK ON!

        Parameters
        ----------
        comm : ExaComm
            Type of comm to test
        """
        acomm = comm(None, 1, 1)
        assert mpi4py.rc.threads != mpi4py_rc
        assert mpi4py.rc.recv_mprobe != mpi4py_rc

    @pytest.mark.parametrize("comm", TestCommHelper.comm_types)
    def test_has_MPI(self, comm):
        """
        Tests that each class has a static member called MPI.
        Comms using MPI should be the first to import mpi4py
        as certain systems require:
            mpi4py.rc.threads = False
            mpi4py.rc.recv_mprobe = False
        When other files need to use the raw MPI than can
        access this or the raw method if they have a ExaComm
        instance.

        Parameters
        ----------
        comm : ExaComm
            Type of comm to test
        """
        assert hasattr(comm, "MPI")
        acomm = comm(None, 1, 1)
        assert isinstance(comm.MPI, type(mpi4py.MPI))
        assert isinstance(comm.MPI.COMM_WORLD, mpi4py.MPI.Comm)

    @pytest.mark.parametrize("comm", TestCommHelper.comm_types)
    def test_raw(self, comm):
        """
        Test the raw member of the ExaComm class.  Raw
        give access to mpi4py's MPI.

        Parameters
        ----------
        comm : ExaComm
            Type of comm to test
        """
        acomm = comm(None, 1, 1)
        assert isinstance(acomm.raw(), mpi4py.MPI.Comm)

    @pytest.mark.parametrize("comm", TestCommHelper.comm_types)
    def test_has_size(self, comm):
        """
        Test the size of a vanilla comm.  Compares against
        the global comm size.  Splits will occur on the
        static members of ExaComm.

        Parameters
        ----------
        comm : ExaComm
            Type of comm to test
        """
        a_comm = comm(None, 1, 1)
        assert hasattr(a_comm, "size")
        assert isinstance(a_comm.size, int)
        assert a_comm.size == mpi4py.MPI.COMM_WORLD.Get_size()

    @pytest.mark.parametrize("comm", TestCommHelper.comm_types)
    def test_has_rank(self, comm):
        """
        Test the rank of a vanilla comm.  Compares against
        the global comm rank.  Splits will occur on the
        static members of ExaComm.

        Parameters
        ----------
        comm : ExaComm
            Type of comm to test
        """
        a_comm = comm(None, 1, 1)
        assert hasattr(a_comm, "rank")
        assert isinstance(a_comm.size, int)
        assert a_comm.rank == mpi4py.MPI.COMM_WORLD.Get_rank()

    @pytest.mark.parametrize("comm", TestCommHelper.comm_types)
    def test_comm_empty_init(self, comm):
        """
        Checks the static member of ExaComm after a
        vanilla comm. We don't check agent_comm and env_comm
        because 1 learner and 1 agent is only valid for 2 rank
        case and we should be running more ranks to test.
        The rest of the parameters are tested in test_comm_split.

        Parameters
        ----------
        comm : ExaComm
            Type of comm to test
        """
        comm(None, 1, 1)
        assert isinstance(ExaComm.global_comm, ExaComm)
        assert hasattr(ExaComm.global_comm, "rank")
        assert hasattr(ExaComm.global_comm, "size")
        assert ExaComm.num_learners == 1
        assert ExaComm.procs_per_env == 1

    @pytest.mark.parametrize("num_learners, procs_per_env", list(TestCommHelper.get_configs()))
    @pytest.mark.parametrize("comm", TestCommHelper.comm_types)
    def test_comm_split(self, comm, num_learners, procs_per_env):
        """
        Ranks should be divided into three parts:
            Learner - Number of learners (continuous ranks)
            Agent - Number of agents (1 per environment group + 1 per learner)
            Environment - Number per agent

        The following is an example of how ranks should be layed
        out assuming 14 ranks (2 Learners, 5 Agents, 4 Environment)
        Rank  0  1  2  3  4  5  6  7  8  9  10 11 12 13
        L     0  1  -  -  -  -  -  -  -  -  -  -  -  -
        A     0  1  2  -  -  -  3  -  -  -  4  -  -  -
        E     -  -  0  1  2  3  0  1  2  3  0  1  2  3

        This is also a valid rank for sync workflow.
        Rank  0  1  2  3
        L     0  -  -  -
        A     0  -  -  -
        E     0  1  2  3

        Parameters
        ----------
        comm : ExaComm
            Type of comm to test
        num_learners : int
            Number of learners to create
        procs_per_env : int
            Number of ranks per env/agent to create
        """
        rank = mpi4py.MPI.COMM_WORLD.Get_rank()
        size = mpi4py.MPI.COMM_WORLD.Get_size()

        if procs_per_env == size:
            assert num_learners == 1, "Only single learner is supported when procs_per_env == global comm size"
        else:
            total_env = procs_per_env * int((size - num_learners) / procs_per_env)
            assert total_env > 0
            assert size == total_env + num_learners, "Invalided configuration"

        # Do the split
        comm(None, procs_per_env, num_learners)

        # Check the static members
        assert ExaComm.num_learners == num_learners
        assert ExaComm.procs_per_env == procs_per_env

        # Check the global comm
        assert isinstance(ExaComm.global_comm, ExaComm)
        assert ExaComm.global_comm.rank == rank
        assert ExaComm.global_comm.size == size

        # Check the learner comm
        if rank < num_learners:
            assert isinstance(ExaComm.learner_comm, ExaComm)
            assert ExaComm.learner_comm.size == num_learners
            assert ExaComm.learner_comm.rank == rank
        else:
            assert ExaComm.learner_comm is None

        # Check the agent comm
        if procs_per_env == size:
            if rank == 0:
                assert isinstance(ExaComm.learner_comm, ExaComm)
                assert ExaComm.agent_comm.size == 1
                assert ExaComm.agent_comm.rank == 0
            else:
                assert ExaComm.agent_comm is None
        else:
            num_agents = num_learners + (size - num_learners) / procs_per_env
            if rank < num_learners:
                assert isinstance(ExaComm.agent_comm, ExaComm)
                assert ExaComm.agent_comm.size == num_agents
                assert ExaComm.agent_comm.rank == rank
            else:
                checked = False
                for i, global_rank in enumerate(range(num_learners, size, procs_per_env)):
                    if rank == global_rank:
                        assert isinstance(ExaComm.agent_comm, ExaComm)
                        assert ExaComm.agent_comm.size == num_agents
                        assert ExaComm.agent_comm.rank == i + num_learners
                        checked = True
                    else:
                        assert ExaComm.learner_comm is None
                        checked = True
                assert checked == True, "Double check on logic failed. Rank " + str(rank) + " was not checked!"

        # Check the env comm
        if procs_per_env == size:
            assert isinstance(ExaComm.env_comm, ExaComm)
            assert ExaComm.env_comm.size == procs_per_env
            assert ExaComm.env_comm.rank == rank
        elif rank < num_learners:
            assert ExaComm.env_comm is None
        else:
            checked = 0
            for global_rank in range(num_learners, size, procs_per_env):
                for j in range(procs_per_env):
                    checked += 1
                    if global_rank + j == rank:
                        assert isinstance(ExaComm.env_comm, ExaComm)
                        assert ExaComm.env_comm.size == procs_per_env
                        assert ExaComm.env_comm.rank == j
            assert checked == size - num_learners, "Double check on logic failed. Rank " + str(rank) + " was not checked!"

    @pytest.mark.parametrize("num_learners, procs_per_env", list(TestCommHelper.get_configs()))
    @pytest.mark.parametrize("comm", TestCommHelper.comm_types)
    def test_comm_is_learner(self, comm, num_learners, procs_per_env):
        """
        Checks the is_learner method of ExaComm and is child classes.
        This uses and alternate approach from split to validate against
        global rank.

        Parameters
        ----------
        comm : ExaComm
            Type of comm to test
        num_learners : int
            Number of learners to create
        procs_per_env : int
            Number of ranks per env/agent to create
        """
        a_comm = comm(None, procs_per_env, num_learners)
        if mpi4py.MPI.COMM_WORLD.Get_rank() < num_learners:
            assert ExaComm.is_learner()
            assert a_comm.is_learner()
        else:
            assert not ExaComm.is_learner()
            assert not a_comm.is_learner()

    @pytest.mark.parametrize("num_learners, procs_per_env", list(TestCommHelper.get_configs()))
    @pytest.mark.parametrize("comm", TestCommHelper.comm_types)
    def test_comm_is_agent(self, comm, num_learners, procs_per_env):
        """
        Checks the is_agent method of ExaComm and is child classes.
        This uses and alternate approach from split to validate against
        global rank.

        Parameters
        ----------
        comm : ExaComm
            Type of comm to test
        num_learners : int
            Number of learners to create
        procs_per_env : int
            Number of ranks per env/agent to create
        """
        rank = mpi4py.MPI.COMM_WORLD.Get_rank()
        size = mpi4py.MPI.COMM_WORLD.Get_size()
        a_comm = comm(None, procs_per_env, num_learners)

        if ExaComm.global_comm.size == procs_per_env:
            if rank == 0:
                assert ExaComm.is_agent()
                assert a_comm.is_agent()
            else:
                assert not ExaComm.is_agent()
                assert not a_comm.is_agent()
        else:
            if rank < num_learners:
                assert ExaComm.is_agent()
                assert a_comm.is_agent()
            elif (rank - num_learners) % procs_per_env == 0:
                assert ExaComm.is_agent()
                assert a_comm.is_agent()
            else:
                assert not ExaComm.is_agent(), str(ExaComm.is_agent())
                assert not a_comm.is_agent(), a_comm.is_agent()

    @pytest.mark.parametrize("num_learners, procs_per_env", list(TestCommHelper.get_configs()))
    @pytest.mark.parametrize("comm", TestCommHelper.comm_types)
    def test_comm_is_actor(self, comm, num_learners, procs_per_env):
        """
        Checks the is_actor method of ExaComm and is child classes.
        An actor is anyone who is not a learner.
        This uses and alternate approach from split to validate against
        global rank.

        Parameters
        ----------
        comm : ExaComm
            Type of comm to test
        num_learners : int
            Number of learners to create
        procs_per_env : int
            Number of ranks per env/agent to create
        """
        rank = mpi4py.MPI.COMM_WORLD.Get_rank()
        size = mpi4py.MPI.COMM_WORLD.Get_size()
        a_comm = comm(None, procs_per_env, num_learners)
        if ExaComm.global_comm.size == procs_per_env:
            assert ExaComm.is_actor()
            assert a_comm.is_actor()
        else:
            if rank < num_learners:
                assert not ExaComm.is_actor()
                assert not a_comm.is_actor()
            else:
                assert ExaComm.is_actor()
                assert a_comm.is_actor()
