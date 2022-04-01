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
    block : list
        Indicates if we should do off-policy (0) or on-policy (1) learning
    priority_scale : list
        Turns on or off priority replay
    batch_frequency : list
        Indicates how often a batch of data should be sent to learner.
        1 sends data each step, -1 sends data each episode.
    """
    workflows = ["simple", "simple_async"] #["sync", "async", "rma"]
    episodes = [1, 10, 100]
    env_steps = [1, 10, 100]
    workflow_steps = [1, 10, 100]
    block = [True, False]
    priority_scale = [0, 1]
    batch_frequency = [1, -1]

    # workflows = ["simple_rma"]
    # episodes = [10]
    # env_steps = [10]
    # workflow_steps = [10]
    # block = [False]
    # priority_scale = [1]
    # batch_frequency = [1]

    def get_configs(workflow):
        """
        This is a generator that spits out configurations of learners, agents, and procs per agent.
        This is used to generate tests for split. 

        Currently, multi-learner configs are turned off
        Parameters
        ----------
        workflow : string
            Name of workflow.  For simple and sync we only give out 1 learner/agent.
            The rest go to env.  For rest we do various combinations of learner/actor.
        
        Returns
        -------
        Pair
            Number of learners and proccesses per environment for comm setup
        """
        size = mpi4py.MPI.COMM_WORLD.Get_size()
        if workflow == "simple" or workflow == "sync":
            yield 1, size
        else:
            # We start at 1 because we have to have at least one learner
            # for num_learners in range(1, size):
            for num_learners in range(1, 2):
                rem = size - num_learners
                # Iterate over all potential procs_per_env counts
                for i in range(0, rem):
                    # Add one since we want the size of the env_count not index
                    procs_per_env = i + 1
                    # Does it fit, then return it
                    if rem % procs_per_env == 0:
                        yield num_learners, procs_per_env

    def get_workflows():
        """
        This function generates combinations of parameters for testing

        Returns
        -------
        List
            Number of learners, processes per environment, workflow name, 
            number of episodes, number of max steps from the environments perspective,
            number of max steps from the workflows perspective, blocking (on/off-policy),
            priority replay, and batch frequency.
        """
        for workflows in TestWorkflowHelper.workflows:
            for episodes in TestWorkflowHelper.episodes:
                for env_steps in TestWorkflowHelper.env_steps:
                    for workflow_steps in TestWorkflowHelper.workflow_steps:
                        for block in TestWorkflowHelper.block:
                            for priority_scale in TestWorkflowHelper.priority_scale:
                                for batch_frequency in TestWorkflowHelper.batch_frequency:
                                    for num_learners, procs_per_env in TestWorkflowHelper.get_configs(workflows):
                                        yield num_learners, procs_per_env, workflows, episodes, env_steps, workflow_steps, block, priority_scale, batch_frequency

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
def get_args(pytestconfig):
    """
    This sets arguments that can be passed to this test.
    on_policy and behind set how far off-policy/behind when training the workflow can be.
    The parser comes from conftest.py.  We require:
        --on_policy - set to 1 to be on policy.  Otherwise set to number to indicate how far off-policy
        to be. -1 just records not asserting. -1 is the default
        --behind - how old of data to train on.  -1 just records not asserting. -1 is default.
    Sleeping is also added to the train and step calls.  There are two sleeps that can be
    turned on (they are off by default).
        --rank_sleep - sleep unique and fixed about base on global rank
        --random_sleep - sleeps for a random time

    To use call pytest ./utest_env.py --on_policy -1 --behind -1 --random_sleep
    
    Parameters
    ----------
    pytestconfig : 
        Hook for pytest argument parser
    """
    WorkflowTestConstants.on_policy = int(pytestconfig.getoption("on_policy"))
    WorkflowTestConstants.behind = int(pytestconfig.getoption("behind"))
    WorkflowTestConstants.rank_sleep = bool(pytestconfig.getoption("rank_sleep"))
    WorkflowTestConstants.random_sleep = bool(pytestconfig.getoption("random_sleep"))
    print("ARGS:", WorkflowTestConstants.on_policy, WorkflowTestConstants.behind, WorkflowTestConstants.rank_sleep, WorkflowTestConstants.random_sleep)

@pytest.fixture(scope="session", params=list(TestWorkflowHelper.get_workflows()))
def init_comm(request):
    """
    This sets up a comm to test agent with.

    Attributes
    ----------
    request : 
        This is the parameter from fixture decorator.  Use request.param to get value.
        Each request.param has a tuple with the Number of learners and Process per environment
        configuration.

    Returns
    -------
    List
        Returns the test parameters
    """
    num_learners, procs_per_env, *rem = request.param
    ExaSimple(procs_per_env=procs_per_env, num_learners=num_learners)
    assert ExaComm.num_learners == num_learners
    assert ExaComm.procs_per_env == procs_per_env
    yield request.param
   
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
def run_params(log_dir, init_comm):
    """
    Attempt to set candle drivers run_params.  We set this up instead of the 
    candle driver.

    Parameters
    ----------
    log_dir :
        Makes sure the first candleDriver.run_params is created
    init_comm : 
        Test parameters used to populate candleDriver.run_params

    Returns
    -------
    List :
        Returns the test parameters
    """
    _, _, workflow_name, episodes, env_steps, workflow_steps, block, priority_scale, batch_frequency = init_comm
    candleDriver.run_params["n_episodes"] = episodes
    candleDriver.run_params["n_steps"] = workflow_steps
    candleDriver.run_params["episode_block"] = block
    candleDriver.run_params["priority_scale"] = priority_scale
    candleDriver.run_params["batch_frequency"] = batch_frequency
    # WorkflowTestConstants.priority_replay = priority_scale
    return init_comm

@pytest.fixture(scope="session")
def registration():
    """
    This fixture registers the fake environment and agent.  Only happens
    once per run (i.e. scope="session").
    """
    gym.envs.registration.register(id=FakeEnv.name, entry_point=FakeEnv)
    exarl.agents.registration.register(id=FakeAgent.name, entry_point=FakeAgent)

@pytest.fixture(scope="session")
def learner(registration, log_dir, get_args, run_params):
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
    _, _, workflow_name, episodes, env_steps, workflow_steps, block, priority_scale, batch_frequency = run_params
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
        total = 0 if not ExaComm.is_actor() else run_learner.env.total_steps
        total = TestWorkflowHelper.reduce_value(total)
        if not ExaComm.global_comm.rank:
            assert total >= WorkflowTestConstants.episodes * steps * ExaComm.procs_per_env

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
        total = 0 if not ExaComm.is_actor() else run_learner.env.total_resets
        total = TestWorkflowHelper.reduce_value(total)
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
        total = 0 if not ExaComm.is_agent() else run_learner.agent._has_data
        total = TestWorkflowHelper.reduce_value(total)
        if not ExaComm.global_comm.rank:
            assert total >= WorkflowTestConstants.episodes * steps

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
        total = 0 if not ExaComm.is_agent() else run_learner.agent._train
        total = TestWorkflowHelper.reduce_value(total)
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
        total = 0 if not ExaComm.is_agent() else run_learner.agent._target_train
        total = TestWorkflowHelper.reduce_value(total)
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
        total = 0 if not ExaComm.is_agent() else run_learner.agent._total_action
        total = TestWorkflowHelper.reduce_value(total)
        if not ExaComm.global_comm.rank:
            assert total >= WorkflowTestConstants.episodes * steps

    def test_priority_replay(self, run_learner):
        """
        Checks to see all indices where updated after a train.
        An agents _weight_loss_check should be empty, otherwise
        the generate_data call that sent indices was never returned.
        We can accept if the last indices were not returned...

        Parameters
        ----------
        run_learner : FakeLearner
            Contains workflow that has already run
        """
        if ExaComm.is_agent() and ExaComm.is_actor():
            assert len(run_learner.agent._weights_loss_check) <= 1

@pytest.mark.skip(reason="This is a test that requires manual tunning")
class TestWorkflowDelays:
    """
    This class is designed to test how a workflow deals with delay.
    We measure delay in the number of models generated by the learner
    (model delay). A workflow can have 3 types of delay:

    1. Data from old models:
        This is related to off-policy learning.  When we get data
        it comes from an actor who does inference with some model.
        This records compares how many models have been generated
        since when this data was created
    
    2. Off-policy learning:
        This is the other side of the 1. The actor records how
        out of data its model is when it is updated with a new
        model by the learner.

    3. Priority replay delay:
        This records how long in model generations did it take
        to receive the indices and weights from the learner
        after training.

    The constants are used to configure the max allowed delays.

    Attributes
    ----------
    max_behind : int
        Max model delay for training on data
    max_off_policy : int
        Max model delay for updating actor's model
    max_delay : int
        Max model delay for updating indices and weights
        for priority replay
    """
    max_behind = 1
    max_off_policy = 1
    max_delay = 1

    def test_oldest_model_data(self, run_learner):
        """
        Checks max model delay for training on data by learner

        Parameters
        ----------
        run_learner : FakeLearner
            Contains workflow that has already run
        """
        if ExaComm.is_learner():
            assert max(run_learner.agent._behind) <= TestWorkflowDelays.max_behind
            

    def test_most_off_policy(self, run_learner):
        """
        Checks max model delay for updating an actor's model

        Parameters
        ----------
        run_learner : FakeLearner
            Contains workflow that has already run
        """
        if ExaComm.is_agent() and ExaComm.is_actor():
            assert max(run_learner.agent._off_policy) <= TestWorkflowDelays.max_off_policy

    def test_max_delayed_priority_update(self, run_learner):
        """
        Checks max model delay for updating indices and weights using priority replay

        Parameters
        ----------
        run_learner : FakeLearner
            Contains workflow that has already run
        """
        if ExaComm.is_agent() and ExaComm.is_actor():
            assert max(run_learner.agent._priority_delay) <= TestWorkflowDelays.max_delay

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