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

@pytest.fixture(scope="function", params=TestAgentHelper.test_envs)
def agent(request, registered_agent, registered_environment, run_params):
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

class TestAgentMembers:

    def test_init(self, agent):
        if ExaComm.is_agent():
            assert agent.is_learner == ExaComm.is_learner()
        else:
            assert agent == None

    def test_get_weights(self):
        """
        Test getting weights from an agent
        """
        pass

    def test_set_weights(self):
        """set target model weights
        """
        pass

    def test_train(self):
        """train the agent
        """
        pass

    def test_action(self):
        """next action based on current state
        """
        pass

    def test_load(self):
        """load weights
        """
        pass

    def test_save(self):
        """save weights
        """
        pass

    def test_has_data(self):
        """return true if agent has experiences from simulation
        """
        pass
