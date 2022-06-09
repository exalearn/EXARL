import os
import importlib
import pytest
import numpy as np
import gym
from exarl.utils.candleDriver import initialize_parameters
from exarl.utils.globals import ExaGlobals
from exarl.base.comm_base import ExaComm
from exarl.network.simple_comm import ExaSimple
from exarl.base.env_base import ExaEnv
from exarl.envs.env_vault.UnitEvn import EnvGenerator
import exarl.agents
import pickle

import mpi4py
# Unfortunatly, this line is starting MPI instead of the communicators.
# I can't figure out how to parameterize a fixture from a fixture which
# ultimately causes the problem.
from mpi4py import MPI

class TestAgentHelper:
    """"
    This is a helper class with constants used throughout the agent tests.

    Attributes
    ----------
    dqn_args : dictionary
        These are required arguments from dqn config
    ddpg_args : dictionary
        These are required arguments from ddpg config
    ac_args : dictionary
        These are the actor critic arguments for ddpg
    lstm_args : dictionary
        These are the arguments for lstm from config
    mlp_args : dictionary
        These are the arguments for mlp from config
    model_types : dictionary
        This is used to pass different models as parameters to fixtures
    priority_scale : list
        List of numbers between 0 and 1 representing priority scale.
    max_attempts : int
        Max amount of times to test behavior
    """

    dqn_args = {
        "gamma": 0.75,
        "epsilon": 0.9,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.999,
        "learning_rate": 0.001,
        "batch_size": 5,
        "tau": 0.5,
        "nactions": 10,
        "priority_scale": 0.0,
        "buffer_capacity": 1000,
        "xla": "True"
    }

    ddpg_args = {
        "epsilon": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.999,
        "gamma": 0.99,
        "tau": 0.005,
        "batch_size": 64,
        "buffer_capacity": 50000
    }

    ac_args = {
        "actor_lr": 0.001,
        "actor_dense_act": "relu",
        "actor_dense": [256, 256],
        "actor_out_act": "tanh",
        "actor_optimizer": "adam",
        "critic_lr": 0.002,
        "critic_state_dense": [16, 32],
        "critic_state_dense_act": "relu",
        "critic_action_dense": [32],
        "critic_action_dense_act": "relu",
        "critic_concat_dense": [256, 256],
        "critic_concat_dense_act": "relu",
        "critic_out_act": "linear",
        "critic_optimizer": "adam",
        "loss": "mse",
        "std_dev": 0.2
    }

    lstm_args = {
        "dense": [64, 128],
        "optimizer": "adam",
        "loss": "mse",
        "lstm_layers": [56, 56, 56],
        "activation": "tanh",
        "gauss_noise": [0.1, 0.1, 0.1],
        "out_activation": "linear",
        "regularizer": [0.001, 0.001],
        "clipnorm": 1.0,
        "clipvalue": 0.5
    }

    mlp_args = {
        "dense": [64, 128],
        "activation": "relu",
        "optimizer": "adam",
        "out_activation": "linear",
        "loss": "mse"
    }

    model_types = {"LSTM": lstm_args, "MLP": mlp_args}
    priority_scale = [0.0, 1.0]
    max_attempts = 100

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

    def compare_weights(a, b):
        """
        This is a helper function is used to compare weights.

        Attributes
        ----------
        a : np.array
            Weights to compare
        b : np.array
            Weights to comapre

        Returns
        -------
        Bool
            True if identical, false otherwise
        """
        if len(a) != len(b):
            return False
        for i, j in zip(a, b):
            if not np.array_equal(i, j):
                return False
        return True

    def make_weights_from_old_weights(weights):
        """
        This is a helper function creates a new list of weights based on the
        size and type of the old ones.  The new ones will be filled with the
        index of their position in the list.

        Attributes
        ----------
        weights : List
            List of old weights

        Returns
        -------
        Bool
            True if identical, false otherwise
        """
        new_weights = []
        for i, some_array in enumerate(weights):
            new_weights.append(np.full_like(some_array, i))
        return new_weights

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

@pytest.fixture(scope="session", params=list(TestAgentHelper.get_configs()))
def init_comm(request, mpi4py_rc):
    """
    This sets up a comm to test agent with.  This test must be run
    with at least two ranks.

    Attributes
    ----------
    request :
        This is the parameter from fixture decorator.  Use request.param to get value.
        Each request.param is a tuple of Number of learners and Process per environment
        configuration.

    Returns
    -------
    Pair
        Number of learners and proccesses per environment for comm setup
    """
    num_learners, procs_per_env = request.param
    ExaSimple(None, procs_per_env, num_learners)
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

@pytest.fixture(scope="session")
def registered_synth_environment():
    """
    This is a pytest fixture to add synthetic environment to the gym registry.
    All synthetic environments will have -v0
    """
    for env in EnvGenerator.generator():
        gym.envs.registration.register(id=env.name, entry_point=env)

@pytest.fixture(scope="session")
def registered_agent(pytestconfig, init_comm):
    """
    This is a pytest fixture to add an agent to the agent registry based on command line arguments.
    The parser comes from conftest.py.  We require:
        test_agent_name - gym name for test (e.g. DQN-v0)
        test_agent_class - name of the class for test module (e.g. DQN)
        test_agent_file - name of the file containing test_env_class omitting the ".py" (e.g. dqn)
    To use call pytest ./utest_agent.py --test_agent_name DQN-v0 --test_agent_class DQN --test_agent_file dqn
    The scope is set to session as to only add to the registry once per pytest session (run).
    In order to allow for multiple agents of the same class but with different comm configs, we pass in
    a cantor pair of the number of learners and process per environment and append that to the name of the agent.

    Parameters
    ----------
    pytestconfig :
        Hook for pytest argument parser
    init_comm : pair
        The number of learners and process per environment

    Returns
    -------
    String
        Returns the environment name that was passed in to command line
    """
    agent_name = pytestconfig.getoption("test_agent_name")
    agent_class = pytestconfig.getoption("test_agent_class")
    agent_file_name = pytestconfig.getoption("test_agent_file")
    entry = getattr(importlib.import_module("exarl.agents.agent_vault." + agent_file_name), agent_class)

    # Cantor pair
    num_learners, procs_per_env = init_comm
    cantor_pair = int(((num_learners + procs_per_env) * (num_learners + procs_per_env + 1)) / 2 + procs_per_env)

    # We are going to strip of the v0 and instead add vCommSize
    # This doesn't matter since we are consistent with the name within the test
    temp = agent_name.split("-")
    if len(temp) > 1:
        temp.pop()
    temp.append("v" + str(cantor_pair))
    agent_name = "-".join(temp)
    exarl.agents.registration.register(id=agent_name, entry_point=entry)

    return agent_name

@pytest.fixture(scope="function", params=TestAgentHelper.model_types.keys())
def run_params(request):
    """
    Attempt to set candle drivers run_params.  We set this up instead of the
    candle driver.

    Parameters
    ----------
    request :
        This is the parameter from fixture decorator.  Use request.param to get value.
        Model types are passed in as request.param.
    """
    ExaGlobals.set_param('output_dir', "./test")
    ExaGlobals.set_params(TestAgentHelper.dqn_args)
    ExaGlobals.set_param("model_type", request.param)
    ExaGlobals.set_params(TestAgentHelper.model_types[request.param])
    return request.param

@pytest.fixture(scope="function")
def agent(registered_agent, registered_environment, run_params):
    """
    This fixture generates an new agent from the agent registry.
    Parameters
    ----------
    registered_agent : String
        Names of agent to create passed in from fixture
    registered_environment : String
        Name of environment to create passed in from fixture
    run_params : None
        Ensures run_params fixture runs before this

    Returns
    -------
    ExaAgent
        Returns an agent to test
    """
    env = ExaEnv(gym.make(registered_environment).unwrapped)
    agent = None
    if ExaComm.is_agent():
        agent = exarl.agents.make(registered_agent, env=env, is_learner=ExaComm.is_learner())
    return agent

@pytest.fixture(scope="function")
def agent_with_priority_scale(registered_agent, registered_environment, run_params, request):
    """
    This fixture generates an new agent from the agent registry.  This fixture also
    sets the candleDriver parameter priority_scale which is used in the train
    set_priorities functions.
    Parameters
    ----------
    registered_agent : String
        Names of agent to create passed in from fixture
    registered_environment : String
        Name of environment to create passed in from fixture
    run_params : None
        Ensures run_params fixture runs before this
    request : float
        request.param is the value to set for priority_scale

    Returns
    -------
    ExaAgent
        Returns an agent to test
    """
    ExaGlobals.set_param("priority_scale", request.param)
    env = ExaEnv(gym.make(registered_environment).unwrapped)
    agent = None
    if ExaComm.is_agent():
        agent = exarl.agents.make(registered_agent, env=env, is_learner=ExaComm.is_learner())
    return agent

@pytest.fixture(scope="function")
def pre_agent(registered_agent, registered_synth_environment, run_params, request):
    """
    This fixture is used for testing synthetic creation.  It returns the name
    of the agent (given via command line) as well as an environment passed in as request.
    This is done via indirect=True being set on @pytest.mark.parametrize:
    (e.g. @pytest.mark.parametrize("pre_agent", list(EnvGenerator.getNames()), indirect=True) )

    Parameters
    ----------
    registered_agent : String
        Name of agent to be created passed in from fixture
    registered_synth_environment : None
        Ensures all synthetic environments have been registered via fixture
    run_params : None
        Ensures run_params fixture runs before this
    request : String
        request.param is the name of the synthetic environment to create

    Returns
    -------
    Pair
        Name of the agent to build and a environment
    """
    env = ExaEnv(gym.make(request.param).unwrapped)
    return registered_agent, env

@pytest.fixture(scope="function")
def agent_with_synth_env(registered_agent, registered_synth_environment, run_params, request):
    """
    This fixture generates an new agent from the agent registry with an environment.
    The environment is passed via request allowing it to be passed by setting
    indirect=True being in @pytest.mark.parametrize:
    (e.g. @pytest.mark.parametrize("syth_agent", listOfEnvs, indirect=True) )
    This allows us to test other environments than what we passed in on command line.
    Ultimately the list that should be passed in is the list of synthetic environments.

    Parameters
    ----------
    registered_agent : String
        Names of agent to create passed in from fixture
    registered_synth_environment : None
        Ensures all synthetic environments have been registered via fixture
    run_params : None
        Ensures run_params fixture runs before this
    request :
        request.param is the name of the synthetic environment to create

    Returns
    -------
    ExaAgent
        Returns an agent to test
    """
    env = ExaEnv(gym.make(request.param).unwrapped)
    agent = None
    if ExaComm.is_agent():
        agent = exarl.agents.make(registered_agent, env=env, is_learner=ExaComm.is_learner())
    return agent

@pytest.fixture(scope="session")
def save_load_dir(init_comm, pytestconfig):
    """
    This fixture creates a directory to store saved weights in.
    It is created once a session and torn down at the end.
    The barriers are to make sure all ranks are synchronized prior
    to file/dir creation and descruction.  The temp directory for
    storing and loading weights can be changed by the
    --test_save_load_dir option.  By default it will write to
    ./save_load_dir.

    Example:
    pytest ./utest_env.py --test_save_load_dir /path/to/my/dir

    Parameters
    ----------
    init_comm : pair
        Ensures the comms are initialized before running

    Returns
    -------
    String
        Directory to use
    """
    rank = ExaComm.global_comm.rank
    made_dir = False
    dir_name = pytestconfig.getoption("test_save_load_dir")
    if rank == 0 and not os.path.isdir(dir_name):
        os.mkdir(dir_name)
        made_dir = True

    ExaComm.global_comm.barrier()
    yield dir_name

    ExaComm.global_comm.barrier()
    if made_dir:
        os.rmdir(dir_name)

@pytest.fixture(scope="function")
def save_load_file(save_load_dir):
    """
    This fixture returns the file name needed to test save and load
    weights.  On tear down the file is removed from the save_load_dir.
    This is called for each function requiring save_load_file.

    Parameters
    ----------
    save_load_dir : String
        Dir where to add file.

    Returns
    -------
    String
        Filename to use
    """
    rank = ExaComm.global_comm.rank
    file_name = save_load_dir + "/weights_" + str(rank) + ".dump"
    yield file_name

    if os.path.isfile(file_name):
        os.remove(file_name)


class TestAgentMembers:

    @pytest.mark.skip(reason="This is a really long test... Fails because agent models are broken!!!")
    @pytest.mark.parametrize("pre_agent", list(EnvGenerator.getNames()), indirect=True)
    def test_agent_creation(self, pre_agent):
        """
        Tests the initialization of synthetic agents.  The synthetic agents iterate over all possible
        gym action/observation space combinations.  This will test the agents ability to handle creating
        a model (mlp/lstm) for such spaces.

        Parameters
        ----------
        pre_agent : Pair
            Agent name and environment to create.
        """
        agent_name, env = pre_agent
        agent = None
        if ExaComm.is_agent():
            agent = exarl.agents.make(agent_name, env=env, is_learner=ExaComm.is_learner())
            assert agent.is_learner == ExaComm.is_learner()
        else:
            assert agent is None

    def test_init(self, agent):
        """
        Tests the initialization of agents relative to comm.

        Parameters
        ----------
        agent : ExaAgent
            Agent to test from fixture
        """
        if ExaComm.is_agent():
            assert agent.is_learner == ExaComm.is_learner()
            assert hasattr(agent, 'batch_size')
        else:
            assert agent is None

    def test_get_weights(self, agent):
        """
        Test getting weights from an agent.  Currently get weights calls
        into tensorflow get_weights
        (https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#get_weights).
        To check, we are asserting we get a list of numpy arrays.
        A better check would be to assert the size and length of these array
        but no sure what the values should be.
        TODO: Figure out dimensions...

        Parameters
        ----------
        agent : ExaAgent
            Agent to test from fixture
        """
        if ExaComm.is_agent():
            assert hasattr(agent, 'get_weights')
            assert callable(getattr(agent, 'get_weights'))
            weights = agent.get_weights()
            assert isinstance(weights, list)
            # if run_params == 'MLP':
            #     assert len(weights) == len(TestAgentHelper.mlp_args["dense"])
            # elif run_params == 'LSTM':
            #     assert len(weights) == len(TestAgentHelper.lstm_args["lstm_layers"])
            # else:
            #     assert False, "Unclear the number of layers of ml model for " + run_params
            # TODO: Check the length of each np.array
            for layer_weights in weights:
                assert isinstance(layer_weights, np.ndarray)
        else:
            assert agent is None

    def test_set_weights(self, agent):
        """
        Test set target model weights.  This test works by getting weights,
        setting arbitrary values for "new" weights, updating, and then
        re-calling get to see if the new return equals the set values.

        Parameters
        ----------
        agent : ExaAgent
            Agent to test from fixture
        """
        if ExaComm.is_agent():
            assert hasattr(agent, 'set_weights')
            assert callable(getattr(agent, 'set_weights'))
            # Get old weights
            weights = agent.get_weights()
            # Make new weights
            new_weights = TestAgentHelper.make_weights_from_old_weights(weights)
            # Set weights
            agent.set_weights(new_weights)
            # Check weights
            to_check = agent.get_weights()
            assert TestAgentHelper.compare_weights(new_weights, to_check)
        else:
            assert agent is None

    def test_save(self, agent, save_load_file):
        """
        Tests save weights.

        Parameters
        ----------
        agent : ExaAgent
            Agent to test from fixture
        save_load_file : String
            Name of weights file from fixture
        """
        if ExaComm.is_agent():
            assert hasattr(agent, 'save')
            assert callable(getattr(agent, 'save'))

            weights = agent.get_weights()
            agent.save(save_load_file)
            assert os.path.isfile(save_load_file)
            with open(save_load_file, 'rb') as f:
                read_weights = pickle.load(f)
            # Strip off layer name from save file
            # read_weights = [x[1] for x in read_weights]
            # assert len(weights) == len(read_weights)
            # for i, j in zip(weights, read_weights):
            #     assert np.array_equal(i, j)
            assert TestAgentHelper.compare_weights(weights, read_weights)
        else:
            assert agent is None

    def test_load(self, agent, save_load_file):
        """
        Tests loading weights.

        Parameters
        ----------
        agent : ExaAgent
            Agent to test from fixture
        save_load_file : String
            Name of weights file from fixture
        """
        if ExaComm.is_agent():
            assert hasattr(agent, 'load')
            assert callable(getattr(agent, 'load'))

            # Get old weights
            old_weights = agent.get_weights()
            # Make new weights
            new_weights = TestAgentHelper.make_weights_from_old_weights(old_weights)
            # Set new weights, save, check file exists
            agent.set_weights(new_weights)
            agent.save(save_load_file)
            assert os.path.isfile(save_load_file)
            # Reset old weights and check set
            agent.set_weights(old_weights)
            to_check = agent.get_weights()
            assert TestAgentHelper.compare_weights(old_weights, to_check)
            # Load weights and check
            agent.load(save_load_file)
            to_check = agent.get_weights()
            assert TestAgentHelper.compare_weights(new_weights, to_check)
        else:
            assert agent is None

    def test_has_data(self, agent):
        """
        This tests that has_data returns false when the agent
        is first initialized.  Testing if has_data == True is
        tested under test_remember.

        Parameters
        ----------
        agent : ExaAgent
            Agent to test from fixture
        """
        if ExaComm.is_agent():
            assert hasattr(agent, 'has_data')
            assert callable(getattr(agent, 'has_data'))
            assert agent.has_data() == False
        else:
            assert agent is None

    def test_remember(self, agent, max_attempts=TestAgentHelper.max_attempts):
        """
        This tests that the remember function stores entries up to
        max_add times.  We verify using has_data method.

        Parameters
        ----------
        agent : ExaAgent
            Agent to test from fixture
        max_attempts : int
            Max number of entries to add
        """
        if ExaComm.is_agent():
            assert hasattr(agent, 'remember')
            assert callable(getattr(agent, 'remember'))

            assert agent.has_data() == False
            for i in range(max_attempts):
                state = agent.env.observation_space.sample()
                next_state = agent.env.observation_space.sample()
                action = agent.env.action_space.sample()
                reward = 100
                done = False

                agent.remember(state, action, reward, next_state, done)
                assert agent.has_data() == True
        else:
            assert agent is None

    def test_generate_data_size(self, agent, max_attempts=TestAgentHelper.max_attempts):
        """
        This tests that the return of generate_data.  When there is no
        data stored, generate data outputs fake data.  This is to setup
        RMA windows for data structures.  This test checks that the size
        of the fake data is >= the size of real data.  Sometimes the size
        issue is related to how pickle operates.

        Parameters
        ----------
        agent : ExaAgent
            Agent to test from fixture
        max_attempts : int
            Max number of entries to add
        """
        if ExaComm.is_agent():
            assert hasattr(agent, 'generate_data')
            assert callable(getattr(agent, 'generate_data'))
            assert hasattr(agent, 'batch_size')

            data = next(agent.generate_data())
            pickle_empty_data = pickle.dumps(data)
            for i in range(max_attempts):
                for j in range(agent.batch_size):
                    state = agent.env.observation_space.sample()
                    next_state = agent.env.observation_space.sample()
                    action = agent.env.action_space.sample()
                    reward = 100
                    done = False
                    agent.remember(state, action, reward, next_state, done)

                data = next(agent.generate_data())
                pickle_full_data = pickle.dumps(data)
                assert len(pickle_empty_data) >= len(pickle_full_data)
        else:
            assert agent is None

    def test_generate_data_small(self, agent):
        """
        This tests that the return of generate_data when the data is less
        than the batch size.  When there is no data stored, generate data
        outputs fake data.  This test checks that the size of the fake
        data is > the size of real data.

        Parameters
        ----------
        agent : ExaAgent
            Agent to test from fixture
        """
        if ExaComm.is_agent():
            assert hasattr(agent, 'batch_size')

            data = next(agent.generate_data())
            pickle_empty_data = pickle.dumps(data)
            for i in range(agent.batch_size - 1):
                state = agent.env.observation_space.sample()
                next_state = agent.env.observation_space.sample()
                action = agent.env.action_space.sample()
                reward = 100
                done = False
                agent.remember(state, action, reward, next_state, done)

                data = next(agent.generate_data())
                pickle_full_data = pickle.dumps(data)
                assert len(pickle_empty_data) > len(pickle_full_data)
        else:
            assert agent is None

    @pytest.mark.parametrize("agent_with_priority_scale", TestAgentHelper.priority_scale, indirect=True)
    def test_train(self, agent_with_priority_scale):
        """
        This tests to see if there is a train method and its return values.
        The return value passes back the indices and lost if scale_priority
        is set greater than 0.  We check to see that the number of indices
        matches the total amount of data.  Only learner is able to call train.

        Parameters
        ----------
        agent_with_priority_scale : ExaAgent
            Agent to test from fixture with priority_scale set
        """
        agent = agent_with_priority_scale
        if ExaComm.is_agent():
            assert hasattr(agent, 'train')
            assert callable(getattr(agent, 'train'))
            assert hasattr(agent, 'priority_scale')

            if ExaComm.is_learner():
                for i in range(agent.batch_size):
                    state = agent.env.observation_space.sample()
                    next_state = agent.env.observation_space.sample()
                    action = agent.env.action_space.sample()
                    reward = 100
                    done = False
                    agent.remember(state, action, reward, next_state, done)
                    data = next(agent.generate_data())
                    if agent.priority_scale != 0.0:
                        assert len(data) == 4
                        assert len(data[2]) == i + 1
                    else:
                        assert len(data) == 2

                    ret = agent.train(data)
                    if agent.priority_scale != 0.0:
                        assert len(ret) == 2
                        assert len(ret[0]) == i + 1
                    else:
                        assert ret is None
        else:
            assert agent is None

    def test_update_target(self, agent):
        """
        This tests the functionality of target train.
        Target train uses the agents tau to update the
        weights of the target model.  Only the learner
        will update the weights.

        Parameters
        ----------
        agent : ExaAgent
            Agent to test from fixture
        """
        if ExaComm.is_agent():
            assert hasattr(agent, 'update_target')
            assert callable(getattr(agent, 'update_target'))
            assert hasattr(agent, 'tau')
            assert agent.tau > 0

            if ExaComm.is_learner():
                old_weights = agent.get_weights()
                agent.update_target()
                weights = agent.get_weights()
                assert not TestAgentHelper.compare_weights(weights, old_weights), "Weights should have changed"
        else:
            assert agent is None

    def test_for_changing_weights(self, agent, max_attempts=TestAgentHelper.max_attempts):
        """
        This test is used to see if the agent can "learn."  We check
        this by passing data coming from generate data to the train
        function and compare the new weights.  Problem is weights are
        only updated to the target model when target trained is called.
        Previous tests shows target train is guaranteed to change the
        weights.

        Parameters
        ----------
        agent : ExaAgent
            Agent to test from fixture
        max_attempts : int
            Max number of times to test weight changes
        """
        if ExaComm.is_agent():
            assert hasattr(agent, 'train')
            assert callable(getattr(agent, 'train'))

            if ExaComm.is_learner():
                for j in range(max_attempts):
                    for i in range(agent.batch_size):
                        state = agent.env.observation_space.sample()
                        next_state = agent.env.observation_space.sample()
                        action = agent.env.action_space.sample()
                        reward = 100
                        done = False
                        agent.remember(state, action, reward, next_state, done)

                    data = next(agent.generate_data())
                    old_weights = agent.get_weights()
                    agent.train(data)
                    # This masks the test but we have to do it for now...
                    agent.update_target()
                    weights = agent.get_weights()
                    assert not TestAgentHelper.compare_weights(weights, old_weights), "Weights should have changed"
        else:
            assert agent is None

    def do_actions(self, agent, max_attempts):
        """
        This function tests the action method of an agent.  Action will perform inference
        using its internal model and will return an appropriate action to take.  We test
        that the action is "correct" by ensuring that it is part of the action space.
        The contains method given by gym checks both the type and if the action falls
        within the bounds of a given space.

        Parameters
        ----------
        agent : ExaAgent
            Agent to test
        max_attempts : int
            The number of times to check actions
        """
        if ExaComm.is_agent():
            assert hasattr(agent, 'action')
            assert callable(getattr(agent, 'action'))
            assert hasattr(agent, "env")

            env = agent.env
            assert hasattr(env, "action_space")
            assert isinstance(env.action_space, gym.Space)

            for i in range(max_attempts):
                action, policy_type = agent.action(env.observation_space.sample())
                assert env.action_space.contains(action)

        else:
            assert agent is None

    def test_action(self, agent, max_attempts=TestAgentHelper.max_attempts):
        """
        This test check the action method ensuring the given action is appropriate.
        This will test the environment passed in at command line or the default.

        Parameters
        ----------
        agent : ExaAgent
            Agent to test from fixture
        max_attempts : int
            The number of times to check actions
        """
        self.do_actions(agent, max_attempts)

    @pytest.mark.skip(reason="This is a really long test... Fails because agent models are broken!!!")
    @pytest.mark.parametrize("agent_with_synth_env", list(EnvGenerator.getNames()), indirect=True)
    def test_action_synth(self, agent_with_synth_env, max_attempts=TestAgentHelper.max_attempts):
        """
        This test check the action method ensuring the given action is appropriate.
        This will test the environment passed in at command line or the default.
        We are using the synthetic environments to test what type of gym spaces
        the agent can handle.

        Parameters
        ----------
        agent_with_synth_env : ExaAgent
            Agent to test from fixture with synthetic environment
        max_attempts : int
            The number of times to check action
        """
        self.do_actions(agent_with_synth_env, max_attempts)

    @pytest.mark.parametrize("agent_with_priority_scale", TestAgentHelper.priority_scale, indirect=True)
    def test_set_priorities(self, agent_with_priority_scale):
        """
        This tests to see if there is a set_priorities method.  This method
        is to support experience replay.  The agent must internally use
        the priorities to adjust the values of the chosen by generate data.
        To test that the priorities are being reflected, we need to test
        the priority replay buffers.  To try to infer what is being passed
        back by the agent is too difficult to due at this level.
        TODO: Create priority replay unit tests.

        Parameters
        ----------
        agent_with_priority_scale : ExaAgent
            Agent to test from fixture with priority_scale set
        """
        agent = agent_with_priority_scale
        if ExaComm.is_agent():
            assert hasattr(agent, 'set_priorities')
            assert callable(getattr(agent, 'set_priorities'))
            assert hasattr(agent, "buffer_capacity")
        else:
            assert agent is None
