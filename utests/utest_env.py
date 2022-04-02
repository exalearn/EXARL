import importlib
import pytest
import numpy as np
import gym
from exarl.base.comm_base import ExaComm
from exarl.network.simple_comm import ExaSimple
from exarl.base.env_base import ExaEnv
from exarl.envs.env_vault.UnitEvn import EnvGenerator
import mpi4py

class TestEnvHelper:
    """"
    This is a helper class with constants used throughout the environment tests.

    Attributes
    ----------
    max_steps : int
        Max allotted steps taken to perform tests
    max_resets : dictionary
        Max allotted resets taken to perform tests
    """
    max_steps = 10000
    max_resets = 10000

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
        size = mpi4py.MPI.COMM_WORLD.Get_size()
        # We start at 1 because we have to have at least one learner
        for num_learners in range(1, size):
            rem = size - num_learners
            # Iterate over all potential procs_per_env counts
            for i in range(0, rem):
                # Add one since we want the size of the env_count not index
                procs_per_env = i + 1
                # Does it fit, then return it
                if rem % procs_per_env == 0:
                    yield num_learners, procs_per_env

    def reset(env):
        """
        This helper resets the environment and checks that the
        resulting state is consistent across ranks.

        Attributes
        ----------
        env : ExaComm
            Environment to reset

        Returns
        -------
        gym.space
            Current observation state after reset
        """
        count = 0
        state = env.reset()
        for i in range(0, ExaComm.env_comm.size):
            to_check = ExaComm.env_comm.bcast(state, i)
            if isinstance(state, np.ndarray):
                if np.array_equal(to_check, state):
                    count += 1
            else:
                if to_check == state:
                    count += 1
        assert count == ExaComm.env_comm.size
        return state

@pytest.fixture(scope="session", params=list(TestEnvHelper.get_configs()))
def init_comm(request):
    """
    This sets up a comm to test environment with.  This test must be run
    with at least two ranks.

    Attributes
    ----------
    env : ExaComm
        Environment to reset

    Returns
    -------
    Pair
        Number of learners and proccesses per environment for comm setup
    """
    num_learners, procs_per_env = request.param
    ExaSimple(procs_per_env=procs_per_env, num_learners=num_learners)
    assert ExaComm.num_learners == num_learners
    assert ExaComm.procs_per_env == procs_per_env
    yield num_learners, procs_per_env

    ExaComm.reset()
    assert ExaComm.global_comm is None
    assert ExaComm.agent_comm is None
    assert ExaComm.env_comm is None
    assert ExaComm.num_learners == 1
    assert ExaComm.procs_per_env == 1

@pytest.fixture(scope="session")
def registered_environment(pytestconfig, init_comm):
    """
    This is a pytest fixture to add an environment to the gym registry based on command line arguments.
    The parser comes from conftest.py.  We require:
        test_env_name - gym name for test (e.g. ExaCartPoleStatic-v0)
        test_env_class - name of the class for test module (e.g. ExaCartpoleStatic)
        test_env_file - name of the file containing test_env_class omitting the ".py" (e.g. ExaCartpoleStatic)
    To use call pytest ./utest_env.py --test_env_name ExaCartPoleStatic-v0 --test_env_class ExaCartpoleStatic --test_env_file ExaCartpoleStatic
    If only test_env_name is given, we assume the environment is already in the gym registry.
    If no arguments are given an synthetic environment is generated.
    The scope is set to session as to only add to the gym registry once per pytest session (run).
    In order to make sure that environments are not re-registered for a given configuration,
    we form a cantor pair from the number of learners and the processes per environment
    https://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function.

    Parameters
    ----------
    pytestconfig :
        Hook for pytest argument parser
    init_comm : pair
        Number of learners and the proccesses per environment coming from init_comm fixture

    Returns
    -------
    String
        Returns the new environment name that was registered
    """
    env_name = pytestconfig.getoption("test_env_name")
    env_class = pytestconfig.getoption("test_env_class")
    env_file_name = pytestconfig.getoption("test_env_file")
    if env_name is not None:
        if env_class is not None and env_file_name is not None:
            entry = getattr(importlib.import_module("exarl.envs.env_vault." + env_file_name), env_class)
        # Assume it is already in the gym registry
        else:
            return env_name
    else:
        entry = EnvGenerator.createClass("Discrete", "Box", False, False, True, 75, 20)
        env_name = entry.name

    # Cantor pair
    num_learners, procs_per_env = init_comm
    cantor_pair = int(((num_learners + procs_per_env) * (num_learners + procs_per_env + 1)) / 2 + procs_per_env)

    # We are going to strip of the v0 and instead add vCommSize
    # This doesn't matter since we are consistent with the name within the test
    temp = env_name.split("-")
    if len(temp) > 1:
        temp.pop()
    temp.append("v" + str(cantor_pair))
    env_name = "-".join(temp)
    gym.envs.registration.register(id=env_name, entry_point=entry)
    return env_name

@pytest.fixture(scope="function")
def environment(registered_environment):
    """
    This fixture generates an new environment from the gym registry.

    Parameters
    ----------
    registered_environment : String
        Names of environment to create passed in from fixture

    Returns
    -------
    ExaEnv
        Returns an environment to test
    """
    return ExaEnv(gym.make(registered_environment).unwrapped)

class TestEnvMembers:
    """
    This class checks an environment has the approapriate memember and methods.
    """

    def test_exa_env(self, environment):
        """
        Checks that environment is of type ExaEnv.

        Parameters
        ----------
        environment : ExaEnv
            Environment from fixture to check.
        """
        assert isinstance(environment, ExaEnv)

    def test_action_space(self, environment):
        """
        Checks that class has a gym action space.

        Parameters
        ----------
        environment : ExaEnv
            Environment from fixture to check.
        """
        assert hasattr(environment, "action_space")
        assert isinstance(environment.action_space, gym.Space)

    def test_observation_space(self, environment):
        """
        Checks that class has a gym observation space.

        Parameters
        ----------
        environment : ExaEnv
            Environment from fixture to check.
        """
        assert hasattr(environment, "observation_space")
        assert isinstance(environment.observation_space, gym.Space)

    def test_reset(self, environment):
        """
        Checks that class has a callable reset function and has the correct return type.

        Parameters
        ----------
        environment : ExaEnv
            Environment from fixture to check.
        """
        assert hasattr(environment, "reset")
        assert callable(getattr(environment, 'reset'))

        ret = environment.reset()
        assert isinstance(ret, type(environment.observation_space.sample()))

    def test_step(self, environment):
        """
        Checks that class has a callable step fuction and has the correct return types.

        Parameters
        ----------
        environment : ExaEnv
            Environment from fixture to check.
        """
        assert hasattr(environment, "step")
        assert callable(getattr(environment, 'step'))

        # Must call reset to use a fresh environment
        environment.reset()
        ret = environment.step(environment.action_space.sample())
        assert len(ret) == 4
        assert isinstance(ret[0], type(environment.observation_space.sample()))
        assert isinstance(ret[1], int) or isinstance(ret[1], float)
        assert isinstance(ret[2], bool)

    def test_env_comm(self, environment, init_comm):
        """
        Checks that class has an env_comm.  This should be the case as
        it is an ExaComm.  The comm is only set when called from MPI.

        Parameters
        ----------
        environment : ExaEnv
            Environment from fixture to check.
        init_comm : Pair
            The number of learners and the processes per environment coming from init_comm fixture
        """
        assert hasattr(environment, "env_comm")
        if ExaComm.is_actor():
            assert isinstance(environment.env_comm, ExaComm)
            assert init_comm[1] == environment.env_comm.size

    def test_base_dir(self, environment):
        """
        Checks that class has an base_dir.  This should be the case as
        it is an ExaComm.  The base_dir is used to store environment specfic results.

        Parameters
        ----------
        environment : ExaEnv
            Environment from fixture to check.
        """
        assert hasattr(environment, "base_dir")

class TestEnvFunctionality:
    """
    This class checks the step and reset methods.
    """

    def test_step(self, environment, max_steps=TestEnvHelper.max_steps):
        """
        Records the initial state after reset and compares the difference after taking a step.
        Will take up to max steps to see if state will change.

        Parameters
        ----------
        environment : ExaEnv
            Environment from fixture to check.
        max_steps : int
            The most step to be taken to attempt a change in state
        """
        if ExaComm.is_actor():
            changed = False
            old = TestEnvHelper.reset(environment)
            for i in range(max_steps):
                new, _, done, _ = environment.step(environment.action_space.sample())
                if isinstance(old, np.ndarray):
                    changed = not np.array_equal(old, new)
                else:
                    changed = old != new
                if changed:
                    break
                # Keep trying after a reset if we didn't already break
                if done:
                    TestEnvHelper.reset(environment)
            assert changed == True, "Did not observe change in " + str(max_steps) + " steps"
        ExaComm.global_comm.barrier()

    def test_reset(self, environment, max_steps=TestEnvHelper.max_steps):
        """
        Records the compares the state after taking one step and after reset.
        Will attempt max_steps to see a change from first reset.

        Parameters
        ----------
        environment : ExaEnv
            Environment from fixture to check.
        max_steps : int
            The most step to be taken to attempt a change in state
        """
        if ExaComm.is_actor():
            TestEnvHelper.reset(environment)
            # Attempt to change the state
            done = False
            count = 0
            while not done and count < max_steps:
                old, _, done, _ = environment.step(environment.action_space.sample())
                count += 1
            # Hit reset and compare
            new = TestEnvHelper.reset(environment)
            if isinstance(old, np.ndarray):
                assert not np.array_equal(old, new),  "Did not observe change on reset np array"
            else:
                assert old != new,  "Did not observe change on reset"
        ExaComm.global_comm.barrier()

    def test_seeds(self, environment, num_resets=TestEnvHelper.max_resets):
        """
        Resets the environment num_resets times and looks for how often initial observation space repeats.

        Parameters
        ----------
        environment : ExaEnv
            Environment from fixture to check.
        num_resets : int
            Max number of resets to check
        """
        if ExaComm.is_actor():
            reset_states = [TestEnvHelper.reset(environment)]
            for i in range(num_resets):
                new = TestEnvHelper.reset(environment)
                if isinstance(new, np.ndarray):
                    if any([np.array_equal(old, new) for old in reset_states]):
                        reset_states.append(new)
                else:
                    if new not in reset_states:
                        reset_states.append(new)
            assert len(reset_states) > 1, "Environment has only one intialization seed after " + str(num_resets) + " resets"
        ExaComm.global_comm.barrier()

    def test_max_steps(self, environment, max_steps=TestEnvHelper.max_steps):
        """
        Test for a max number of steps for a given environment.  This looks for an environment
        to return that it is finished by taking random actions up until some theshold.

        Parameters
        ----------
        environment : ExaEnv
            Environment from fixture to check.
        max_steps : int
            Max number of steps to check
        """
        if ExaComm.is_actor():
            end_step = max_steps
            TestEnvHelper.reset(environment)
            for i in range(max_steps):
                _, _, done, _ = environment.step(environment.action_space.sample())
                if done:
                    end_step = i
                    break

            assert end_step < max_steps, "Did not encounter a done state after " + str(max_steps) + " steps"
        ExaComm.global_comm.barrier()
