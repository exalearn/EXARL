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
        ExaEnv(EnvGenerator.createClass("Discrete", "Discrete", False, False, True, 100, 10))
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

@pytest.fixture(scope="session", params=TestAgentHelper.model_types.keys())
def run_params(request):
    """
    Attempt to set candle drivers run_params.
    """
    candleDriver.run_params = {'output_dir' : "./test"}
    candleDriver.run_params.update(TestAgentHelper.dqn_args)
    candleDriver.run_params["model_type"] = request.param
    candleDriver.run_params.update(TestAgentHelper.model_types[request.param])

    # candleDriver.run_params['xla'] = 'True'
    # candleDriver.run_params['gamma']
    # candleDriver.run_params['epsilon']
    # candleDriver.run_params['epsilon_min']
    # candleDriver.run_params['epsilon_decay']
    # candleDriver.run_params['learning_rate']
    # candleDriver.run_params['batch_size']
    # candleDriver.run_params['tau']
    # candleDriver.run_params['model_type']
    # candleDriver.run_params['dense']
    # candleDriver.run_params['lstm_layers']
    # candleDriver.run_params['gauss_noise']
    # candleDriver.run_params['regularizer']
    # candleDriver.run_params['clipnorm']
    # candleDriver.run_params['clipvalue']
    # candleDriver.run_params['activation']
    # candleDriver.run_params['out_activation']
    # candleDriver.run_params['optimizer']
    # candleDriver.run_params['loss']
    # candleDriver.run_params['nactions']
    # candleDriver.run_params['priority_scale']

@pytest.fixture(scope="session", params=list(TestAgentHelper.get_configs()))
def init_comm(request, run_params):
    """
    This sets up a comm to test environment with.  This test must be run
    with at least two ranks.

    Attributes
    ----------
    env : ExaComm
        Environment to reset
    """
    num_learners, procs_per_env = request.param
    ExaSimple(procs_per_env=procs_per_env, num_learners=num_learners)
    assert ExaComm.num_learners == num_learners
    assert ExaComm.procs_per_env == procs_per_env
    yield procs_per_env
   
    ExaComm.reset()
    assert ExaComm.global_comm == None
    assert ExaComm.agent_comm == None
    assert ExaComm.env_comm == None
    assert ExaComm.num_learners == 1
    assert ExaComm.procs_per_env == 1

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
    Parameters
    ----------
    pytestconfig : 
        Hook for pytest argument parser

    Returns
    -------
    String
        Returns the environment name that was passed in to command line
    """
    
    
    agent_name = pytestconfig.getoption("test_agent_name")
    agent_class = pytestconfig.getoption("test_agent_class")
    agent_file_name = pytestconfig.getoption("test_agent_file")
    entry = getattr(importlib.import_module("exarl.agents.agent_vault." + agent_file_name), agent_class)
    
    # We are going to strip of the v0 and instead add vCommSize
    # This doesn't matter since we are consistent with the name within the test
    temp = agent_name.split("-")
    if len(temp) > 1:
        temp.pop()
    temp.append("v"+str(init_comm))
    agent_name = "-".join(temp)
    exarl.agents.registration.register(id=agent_name, entry_point=entry)
    return agent_name

@pytest.fixture(scope="function", params=TestAgentHelper.test_envs)
def agent(request, registered_agent):
    """
    This fixture generates an new agent from the agent registry.
    Parameters
    ----------
    registered_agent : 
        Names of agent to create passed in from fixture

    Returns
    -------
    ExaAgent
        Returns an agent to test
    """
    agent = None
    if ExaComm.is_agent():
        agent = exarl.agents.make(registered_agent, env=request.param, is_learner=ExaComm.is_learner())
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
