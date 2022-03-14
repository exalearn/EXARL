import os
import pytest
import gym
from gym import spaces
from exarl.utils import candleDriver
from exarl.base.comm_base import ExaComm
from exarl.network.simple_comm import ExaSimple
from exarl.base.env_base import ExaEnv
from exarl.base.agent_base import ExaAgent
from exarl.envs.env_vault.UnitEvn import EnvGenerator
import exarl.agents
from .workflow import FakeLearner
from .workflow import FakeAgent
from .workflow import FakeEnv
from .workflow import WorkflowTestConstants
from .workflow import record
import mpi4py

class TestWorkflowHelper:
    """"
    This is a helper class with constants used throughout the workflow tests.
    
    Attributes
    ----------
    workflows : list
        These are the workflows to test
    episodes : list
        The number of episodes to test with
    env_steps : list
        The cut off for the max steps built into the environment to test
    workflow_steps : list
        The max steps per episode set by the workflow to test
    """
    workflows = ["sync"] #["sync", "async", "rma"]
    episodes = [1, 10, 100]
    env_steps = [1, 10, 100]
    workflow_steps = [1, 10, 100]

    def get_workflows():
        """
        This function generates combinations of workflows, episodes, and steps.
        """
        for workflows in TestWorkflowHelper.workflows:
            for episodes in TestWorkflowHelper.episodes:
                for env_steps in TestWorkflowHelper.env_steps:
                    for workflow_steps in TestWorkflowHelper.workflow_steps:
                        yield workflows, episodes, env_steps, workflow_steps

    def get_configs():
        """
        This is a generator that spits out configurations of learners, agents, and procs per agent.
        This is used to generate tests for split. 

        Returns
        -------
        Pair
            Number of learners and proccesses per environment for comm setup
        """
        size = mpi4py.MPI.COMM_WORLD.Get_size()
        if size > 1:
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
        else:
            yield 1, 1

    def reduce_value(value):
        """
        This function is used to aggregate a single value from multiple
        counters on multiple ranks.  Values are aggregated on rank 0.

        Parameters
        ----------
        value : int
            Counter to aggregate
        """
        total = value
        if ExaComm.global_comm.rank:
            ExaComm.global_comm.send(value, 0)
        else:
            data = None
            for i in range(1, ExaComm.global_comm.size):
                data = ExaComm.global_comm.recv(data, source=i)
                total += data
        return total

@pytest.fixture(scope="session")
def set_on_policy(pytestconfig):
    """
    This sets the how far off-policy/behind when training the workflow can be.
    The parser comes from conftest.py.  We require:
        --on_policy - set to 1 to be on policy.  Otherwise set to number to indicate how far off-policy
        to be. Default is 1 (i.e. on-policy).
        --behind - how old of data to train on.  Default is 0.
    To use call pytest ./utest_env.py --on_policy 1 --behind 0
    
    Parameters
    ----------
    pytestconfig : 
        Hook for pytest argument parser
    """
    WorkflowTestConstants.on_policy = int(pytestconfig.getoption("on_policy"))
    WorkflowTestConstants.behind = int(pytestconfig.getoption("behind"))

@pytest.fixture(scope="session", params=list(TestWorkflowHelper.get_configs()))
def init_comm(request):
    """
    This sets up a comm to test agent with.

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
def log_dir(init_comm):
    """
    This fixture creates a directory to store workflow logs in.
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
    dir_name = './log_dir'
    if rank == 0 and not os.path.isdir(dir_name):
        os.mkdir(dir_name)
        made_dir = True
    
    candleDriver.run_params = {'output_dir' : dir_name}
    ExaComm.global_comm.barrier()
    yield dir_name

    ExaComm.global_comm.barrier()
    if made_dir:
        os.rmdir(dir_name)

@pytest.fixture(scope="session")
def registration():
    """
    This fixture registers the fake environment and agent.  Only happens
    once per run (i.e. scope="session").
    """
    gym.envs.registration.register(id=FakeEnv.name, entry_point=FakeEnv)
    exarl.agents.registration.register(id=FakeAgent.name, entry_point=FakeAgent)

@pytest.fixture(scope="session", params=list(TestWorkflowHelper.get_workflows()))
def learner(init_comm, registration, log_dir, request, set_on_policy):
    """
    This fixture generates an new workflow from the workflow registry.
    
    Parameters
    ----------
    init_comm : pair
        Number of learners and the proccesses per environment coming from init_comm fixture
    registration : None
        Ensures the registration fixture has run and we can load fake env and agent

    Returns
    -------
    FakeLearner
        The fake learner containing the env, agent, and workflow
    """
    workflow_name, episodes, env_steps, workflow_steps = request.param
    WorkflowTestConstants.episodes = episodes
    WorkflowTestConstants.env_max_steps = env_steps
    WorkflowTestConstants.workflow_max_steps = workflow_steps
    record.reset()

    env = None
    agent = None
    if ExaComm.is_actor():
        env = ExaEnv(gym.make(FakeEnv.name).unwrapped)
    if ExaComm.is_agent():
        agent = exarl.agents.make(FakeAgent.name, env=env, is_learner=ExaComm.is_learner())
    workflow = exarl.workflows.make(workflow_name)
    return FakeLearner(episodes, workflow_steps, agent, env, workflow, log_dir)

@pytest.fixture(scope="session")
def steps(learner):
    """
    This fixture returns the maximum steps per episode allowed.
    This is so we don't have to recalculate it all the time.

    Parameters
    ----------
    learner : FakeLearner
        From fixture, ensures WorkflowTestConstants are set.

    Returns
    -------
    int
        Steps per episode
    """
    return min(WorkflowTestConstants.env_max_steps, WorkflowTestConstants.workflow_max_steps)

@pytest.fixture(scope="session")
def run_learner(learner):
    """
    This fixture returns a Fake Learner after its ran.
    It only returns a single one per session.

    Parameters
    ----------
    learner : FakeLearner
        From fixture, ensures WorkflowTestConstants are set.

    Returns
    -------
    FakeLearner
        The fake learner after its already run
    """
    ExaComm.global_comm.barrier()
    learner.run()
    ExaComm.global_comm.barrier()
    return learner

class TestWorkflowEnv:
    """
    This is a class of tests that looks at the environment counters.
    """
    def test_steps_per_rank(self, run_learner, steps):
        """
        Checks if all actors have run at least one step. 

        Parameters
        ----------
        run_learner : FakeLearner
            Contains workflow that has already run
        """
        if ExaComm.is_actor():
            assert run_learner.env.total_steps > 0
            assert run_learner.env.total_steps <= WorkflowTestConstants.episodes * steps

    def test_steps(self, run_learner, steps):
        """
        Checks if total steps is equal to the episodes * steps 

        Parameters
        ----------
        run_learner : FakeLearner
            Contains workflow that has already run
        steps : int
            Maximum steps per episode
        """
        total = TestWorkflowHelper.reduce_value(run_learner.env.total_steps)
        if not ExaComm.global_comm.rank:
            assert total == WorkflowTestConstants.episodes * steps * ExaComm.procs_per_env

    def test_reset_per_rank(self, run_learner):
        """
        Checks if all actors reset their env at least once

        Parameters
        ----------
        run_learner : FakeLearner
            Contains workflow that has already run
        """
        if ExaComm.is_actor():
            assert run_learner.env.total_resets > 0
    
    def test_reset(self, run_learner):
        """
        Checks if the total number of resets is at least the number of episodes

        Parameters
        ----------
        run_learner : FakeLearner
            Contains workflow that has already run
        """
        total = TestWorkflowHelper.reduce_value(run_learner.env.total_resets)
        if not ExaComm.global_comm.rank:
            assert total >= WorkflowTestConstants.episodes

class TestWorkflowAgent:
    """
    This is a class of tests that looks at the agent counters.
    """
    def test_has_data(self, run_learner, steps):
        """
        Checks if the total number of remember calls is equal to the total steps

        Parameters
        ----------
        run_learner : FakeLearner
            Contains workflow that has already run
        steps : int
            Maximum steps per episode
        """
        total = TestWorkflowHelper.reduce_value(run_learner.agent._has_data)
        if not ExaComm.global_comm.rank:
            assert total == WorkflowTestConstants.episodes * steps * ExaComm.procs_per_env

    def test_train_per_learner(self, run_learner):
        """
        Checks if each learner calls train

        Parameters
        ----------
        run_learner : FakeLearner
            Contains workflow that has already run
        """
        if ExaComm.is_learner():
            assert run_learner.agent._train > 0

    def test_train(self, run_learner):
        """
        Checks if the total number of train calls is at least the number of episodes

        Parameters
        ----------
        run_learner : FakeLearner
            Contains workflow that has already run
        """
        total = TestWorkflowHelper.reduce_value(run_learner.agent._train)
        if not ExaComm.global_comm.rank:
            assert total >= WorkflowTestConstants.episodes
    
    def test_target_train_per_learner(self, run_learner):
        """
        Checks if each learner calls target_train

        Parameters
        ----------
        run_learner : FakeLearner
            Contains workflow that has already run
        """
        if ExaComm.is_learner():
            assert run_learner.agent._target_train > 0

    def test_target_train(self, run_learner):
        """
        Checks if the total number of target_train calls is at least the number of episodes

        Parameters
        ----------
        run_learner : FakeLearner
            Contains workflow that has already run
        """
        total = TestWorkflowHelper.reduce_value(run_learner.agent._target_train)
        if not ExaComm.global_comm.rank:
            assert total >= WorkflowTestConstants.episodes

    def test_action_per_learner(self, run_learner):
        """
        Checks if the actions called on each agent with env comm = 0

        Parameters
        ----------
        run_learner : FakeLearner
            Contains workflow that has already run
        """
        if ExaComm.is_agent() and ExaComm.is_actor():
            assert run_learner.agent._total_action > 0

    def test_action(self, run_learner, steps):
        """
        Checks if all the actions called is at least the number of episodes * steps

        Parameters
        ----------
        run_learner : FakeLearner
            Contains workflow that has already run
        """
        total = TestWorkflowHelper.reduce_value(run_learner.agent._total_action)
        if not ExaComm.global_comm.rank:
            assert total >= WorkflowTestConstants.episodes * steps


# Notes on things to check:

# Number of steps
# Number of episodes
# Data passed back and forth matches
# Is the data passed between actor and learner correct
# Does everyone in the env comm do the same action
# Is the environment reset
# Is the state updated

# Call Pattern
# How often are the weights updated
# Do they have current_state, total_reward
# Epsilon???