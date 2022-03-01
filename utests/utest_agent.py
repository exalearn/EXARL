import os
import importlib
import pytest
import numpy as np
import gym
from exarl.utils import candleDriver
from exarl.base.comm_base import ExaComm
from exarl.network.simple_comm import ExaSimple
from exarl.base.env_base import ExaEnv
from exarl.envs.env_vault.UnitEvn import EnvGenerator
import exarl.agents
import mpi4py
import pickle

class TestAgentHelper:
    """"
    This is a helper class with constants used throughout the agent tests.
    
    Attributes
    ----------
    test_envs : List
        
    """
    test_envs = [
        EnvGenerator.createClass("Discrete", "Discrete", False, False, True, 100, 10)
    ]

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
        "mem_length": 1000,
        "xla": "True"
    }

    ddpg_args = {
        "epsilon": 1.0,
        "epsilon_min" : 0.01,
        "epsilon_decay" : 0.999,
        "gamma": 0.99,
        "tau" : 0.005,
        "batch_size" : 64,
        "buffer_capacity": 50000
    }

    ac_args = {
        "actor_lr" : 0.001,
        "actor_dense_act" : "relu",
        "actor_dense": [256, 256],
        "actor_out_act" : "tanh",
        "actor_optimizer" : "adam",
        "critic_lr" : 0.002,
        "critic_state_dense": [16, 32],
        "critic_state_dense_act" : "relu",
        "critic_action_dense": [32],
        "critic_action_dense_act" : "relu",
        "critic_concat_dense": [256, 256],
        "critic_concat_dense_act" : "relu",
        "critic_out_act" : "linear",
        "critic_optimizer" : "adam",
        "loss" : "mse",
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
        "dense" : [64, 128],
        "activation" : "relu",
        "optimizer" : "adam",
        "out_activation" : "linear",
        "loss" : "mse"
    }

    model_types = {"LSTM" : lstm_args, "MLP" : mlp_args}

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

@pytest.fixture(scope="session", params=list(TestAgentHelper.get_configs()))
def init_comm(request):
    """
    This sets up a comm to test agent with.  This test must be run
    with at least two ranks.  The value returned is a cantor pair (i.e. unique number)
    which can be used to (re-) register agents within the agent registry.

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
    ExaSimple(procs_per_env=procs_per_env, num_learners=num_learners)
    assert ExaComm.num_learners == num_learners
    assert ExaComm.procs_per_env == procs_per_env
    yield num_learners, procs_per_env
   
    ExaComm.reset()
    assert ExaComm.global_comm == None
    assert ExaComm.agent_comm == None
    assert ExaComm.env_comm == None
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
    temp.append("v"+str(cantor_pair))
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
    init_comm : int
        This is a unique number from comm init used for registering new agents.
    run_params : None
        This is just to make sure the candle driver is run before registering the agent

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
    temp.append("v"+str(cantor_pair))
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
    candleDriver.run_params = {'output_dir' : "./test"}
    candleDriver.run_params.update(TestAgentHelper.dqn_args)
    candleDriver.run_params["model_type"] = request.param
    candleDriver.run_params.update(TestAgentHelper.model_types[request.param])
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
def pre_agent(registered_agent, registered_synth_environment, run_params, request):
    """
    This fixture is used for testing synthetic creation.  It returns the name
    of the command line agent as well as an environment passed in as request.
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
    
    request :
        request.param is the name of the synthetic environment to create

    Returns
    -------
    Pair
        Name of the agent to build and a environment
    """

    env = ExaEnv(gym.make(request.param).unwrapped)
    return registered_agent, env

@pytest.fixture(scope="session")
def save_load_dir(init_comm):
    """
    This fixture creates a directory to store saved weights in.
    It is created once a session and torn down at the end.
    The barriers are to make sure all ranks are synchronized prior
    to file/dir creation and descruction.

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
    dir_name = './save_load_dir'
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
        Tests the initalization of synthetic agents.  The synthetic agents iterate over all possible 
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
        Tests the initalization of agents relative to comm.
        This test should also 

        Parameters
        ----------
        agent : ExaAgent
            Agent to test from fixture
        """
        if ExaComm.is_agent():
            assert agent.is_learner == ExaComm.is_learner()
            assert hasattr(agent, 'batch_size')
        else:
            assert agent == None

    def test_get_weights(self, agent, run_params):
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
            
            # Check the length of each np.array
            for layer_weights in weights:
                assert isinstance(layer_weights, np.ndarray)
        else:
            assert agent == None

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
            # Set weigthts    
            agent.set_weights(new_weights)
            # Check weights
            to_check = agent.get_weights()
            assert TestAgentHelper.compare_weights(new_weights, to_check)
        else:
            assert agent == None

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
            assert agent == None

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
            assert agent == None

    def test_has_data(self, agent):
        """
        This tests that has_data returns false when the agent
        is first initialized.  Testing if has_data == True is 
        tested under test_remember
        
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
            assert agent == None
        

    def test_remember(self, agent, max_add=100):
        """
        This tests that the remember function stores entries up to 
        max_add times.  We verify using has_data method.
        
        Parameters
        ----------
        agent : ExaAgent
            Agent to test from fixture
        max_add : int
            Max number of entries to add
        """
        if ExaComm.is_agent():
            assert hasattr(agent, 'remember')
            assert callable(getattr(agent, 'remember'))
            assert agent.has_data() == False
            
            for i in range(max_add):
                state = agent.env.observation_space.sample()
                next_state = agent.env.observation_space.sample()
                action = agent.env.action_space.sample()
                reward = 100
                done = False
            
                agent.remember(state, action, reward, next_state, done)
                assert agent.has_data() == True
        else:
            assert agent == None

    def test_generate_data_size(self, agent, max_add=100):
        """
        This tests that the return of generate_data.  When there is no
        data stored, generate data outputs fake data.  This is to setup
        RMA windows for data structures.  This test checks that the size
        of the fake data is >= the size of real data.  Sometimes the size
        issue is related to how pickle opperates.
        
        Parameters
        ----------
        agent : ExaAgent
            Agent to test from fixture
        max_add : int
            Max number of entries to add
        """
        if ExaComm.is_agent():
            assert hasattr(agent, 'generate_data')
            assert callable(getattr(agent, 'generate_data'))
            assert agent.has_data() == False
            data = next(agent.generate_data())
            pickle_empty_data = pickle.dumps(data)
            for i in range(max_add):
                for j in range(agent.batch_size):
                    state = agent.env.observation_space.sample()
                    next_state = agent.env.observation_space.sample()
                    action = agent.env.action_space.sample()
                    reward = 100
                    done = False
                    agent.remember(state, action, reward, next_state, done)
                    assert agent.has_data() == True
                data = next(agent.generate_data())
                pickle_full_data = pickle.dumps(data)
                assert len(pickle_empty_data) >= len(pickle_full_data)
        else:
            assert agent == None

    def test_generate_data(self, agent, max_add=100):
        """
        This tests that the return of generate_data.  We are checking the amount
        of experiences that are put into the agent vs how many they get back. 
        
        Parameters
        ----------
        agent : ExaAgent
            Agent to test from fixture
        max_add : int
            Max number of entries to add
        """
        if ExaComm.is_agent():
            assert hasattr(agent, 'generate_data')
            assert callable(getattr(agent, 'generate_data'))
            assert agent.has_data() == False
            
            push_count = 0
            pop_count = 0
            count = 0
            for i in range(max_add):
                for j in range(i):
                    state = agent.env.observation_space.sample()
                    next_state = agent.env.observation_space.sample()
                    action = agent.env.action_space.sample()
                    reward = 100
                    done = False
                
                    agent.remember(state, action, reward, next_state, done)
                    assert agent.has_data() == True
                    push_count += 1
                    count += 1
                while agent.has_data() and count > -1:
                    # TODO: How do we evaluate the return value of generate data
                    ret = agent.generate_data()
                    to_add = 0
                    for k in ret:
                        if to_add == 0:
                            to_add = len(k)
                        assert len(k) == to_add
                    assert to_add <= agent.batch_size
                    pop_count += to_add
                    count -= to_add
                assert count == 0
            assert push_count == pop_count
        else:
            assert agent == None

    def test_train(self):
        """
        train the agent
        """
        pass

    def test_action(self):
        """
        next action based on current state
        """
        pass

    def test_set_priorities(self):
        pass