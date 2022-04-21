import os
import sys
import gym
from gym import spaces
from exarl.utils.candleDriver import initialize_parameters
from exarl.utils.globals import ExaGlobals
from exarl.base.comm_base import ExaComm
from exarl.network.simple_comm import ExaSimple
from exarl.base.env_base import ExaEnv
from exarl.base.agent_base import ExaAgent
from exarl.envs.env_vault.UnitEvn import EnvGenerator
import exarl.agents
import functools
import time
import random

# We fix the seed for repeatablish sleeping
random.seed(7)

class record:
    """
    This class is a helper to replay when something goes wrong
    with a fake environment and agent.

    Attributes
    ----------
    counters : Dictionary
        This contains a name of each function and how many times it ran
    events : List
        This contains a list of events that have occurred on a single node
    verbose : bool
        Flag indicating to print on each event
    """
    counters = {}
    events = []

    verbose = False

    def reset(verbose=False):
        """
        Resets the record.

        Parameters
        ----------
        verbose : bool
            Flag indicating to print on each event
        """
        record.counters = {}
        record.events = []
        record.verbose = verbose

    def event(func):
        """
        This is a decorator to put ontop of a function.  It will record
        how many times it has been call in the counters dictionary and
        add an entry under events.

        Parameters
        ----------
        func : function
            This is the function to record

        Returns
        -------
        result
            Returns the result of the function
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if func.__name__ in record.counters:
                record.counters[func.__name__] += 1
            else:
                record.counters[func.__name__] = 1
            if record.verbose:
                print("[STAND-ALONE WORKFLOW TEST]", ExaComm.global_comm.rank, record.counters[func.__name__], func.__name__, flush=True)
            record.events.append((ExaComm.global_comm.rank, record.counters[func.__name__], func.__name__,))
            result = func(*args, **kwargs)
            return result
        return wrapper


class WorkflowTestConstants:
    """
    This is a helper class/namespace to add a constants set by the workflow
    for the fake environment.  Environment creation goes through gym and
    it is easier to just set a global.

    Attributes
    ----------
    episodes : int
        The maximum number of episodes
    env_max_steps : int
        The maximum number of steps set in the environment
    workflow_max_steps : int
        The maximum number of steps set in the workflow
    on_policy : int
        How much delay an agent can tolerate.  Set to -1
        to ignore assert.
    behind : int
        How old data can be to use to train. Set to -1
        to ignore assert.
    rank_sleep : bool
        Flag to turn on sleeping in step and train based on
        rank
    random_sleep : bool
        Flag to turn on sleeping in step and train based on
        a random amount
    """
    episodes = None
    env_max_steps = None
    workflow_max_steps = None
    on_policy = -1
    behind = -1
    rank_sleep = False
    random_sleep = True

    def do_random_sleep():
        """
        This function look at the constants and performs a sleep
        for some amount of microseconds
        """
        if WorkflowTestConstants.rank_sleep > 0:
            time.sleep(ExaComm.global_comm.rank * 10**(-3))
        elif WorkflowTestConstants.random_sleep > 0:
            time.sleep(int(random.random() * 100) * 10**(-4))

class FakeLearner:
    """
    This class is used to fake out a learner base.  It seems all we really
    need is something that can link the workflow, agent, and environment
    together plus the number of episodes and steps.

    Attributes
    ----------
    global_comm : ExaComm
        The global comm
    global_size : int
        Size of the global comm
    nepisodes : int
        Number of total episodes to run
    nsteps : int
        Number of max steps per episode
    agent : ExaAgent
        Fake agent to use for testing
    env : ExaEnv
        Fake environment to use for testing
    workflow : ExaWorkflow
        Workflow to test
    results_dir : String
        Directory to put logs into
    action_type : int
        Used to indicate what actions to take
    """
    def __init__(self, nepisodes, nsteps, agent, env, workflow, results_dir):
        """
        Parameters
        ----------
        nepisodes : int
            Total number of episodes to run
        nsteps : int
            Max number of steps per episode to run
        agent : ExaAgent
            Fake agent used for testing
        env : ExaEnv
            Fake environment used for testing
        workflow : ExaWorkflow
            The environment to test
        results_dir : String
            Directory where logs are written
        """
        # Doubt we actually need self.global_comm and self.global_size
        self.global_comm = ExaComm.global_comm
        self.global_size = ExaComm.global_comm.size
        self.nepisodes = nepisodes
        self.nsteps = nsteps
        self.agent = agent
        self.env = env
        self.workflow = workflow
        self.results_dir = results_dir
        self.action_type = None

    def run(self):
        """
        Runs the workflow to test.
        """
        self.workflow.run(self)

    def print_delays(self):
        """
        This function prints out statistics about the model
        delays observed by the agent.
        """
        if ExaComm.is_learner() and len(self.agent._behind):
            print("Learner Rank:Min:Max:Ave", ExaComm.global_comm.rank,
                  min(self.agent._behind), max(self.agent._behind), sum(self.agent._behind) / len(self.agent._behind))
        if ExaComm.is_agent() and len(self.agent._off_policy):
            print("Agent Rank:Min:Max:Ave", ExaComm.global_comm.rank,
                  min(self.agent._off_policy), max(self.agent._off_policy), sum(self.agent._off_policy) / len(self.agent._off_policy))
        if ExaComm.is_agent() and len(self.agent._priority_delay):
            print("Agent Priority Rank:Min:Max:Ave", ExaComm.global_comm.rank,
                  min(self.agent._priority_delay), max(self.agent._priority_delay), sum(self.agent._priority_delay) / len(self.agent._priority_delay))

class FakeEnv(gym.Env):
    """
    This is a fake environment.  We use it in coordination with the
    fake agent to ensure that the correct data is passed between
    the environment and the learner.  The state returned from this
    environment always starts at zero (from reset).  State increases
    by one each time until it sets done.  We assert that the action
    is increasing by one at the same rate as the state.  We also
    assert that the step is not called when the environment is done.

    Attributes
    ----------
    name : string
        Name of the fake environment
    action_space : gym space
        Env's action space. We set this to discrete for counting.
        It shouldn't matter that we only test one type.  Space type
        testing is done in agent tests.
    observation_space : gym space
        Env's observation space. We set this to discrete for counting.
        It shouldn't matter that we only test one type.  Space type
        testing is done in agent tests.
    done : bool
        Indicates if the environment is done based on max_steps of the environment.
    state : int
        The current state of the environment.  Is int because we are using discrete space.
    max_steps : int
        The max steps the environment can take.
    total_step : ExaEnv
        Total steps the environment has performed
    total_reset : ExaWorkflow
        The total number or resets
    """

    name = "FakeEnv-v0"

    def __init__(self):
        super().__init__()
        # We allow it to be well over max_steps to see if there will be error
        self.action_space = spaces.Discrete(WorkflowTestConstants.env_max_steps * 10)
        self.observation_space = spaces.Discrete(WorkflowTestConstants.env_max_steps * 10)

        # Init env in bad state to check
        # that workflow call reset first!
        self.done = True
        self.state = WorkflowTestConstants.env_max_steps
        self.max_steps = WorkflowTestConstants.env_max_steps

        self.total_steps = 0
        self.total_resets = 0

    @record.event
    def step(self, action):
        """
        This is a single step.  If number of calls == max steps
        done is set.  State and total_steps are always incremented by 1.
        The action should match the state.  Actions are dictated by the
        inference done on an agent.  This needs to be propageted by
        the head of the env comm.  The state will always just increase
        by one.

        Parameters
        ----------
        action : int
            This is the step to perform. Its an int since the action
            space is Discrete.

        Returns
        -------
        Pair
            Next state, reward, done, and info
        """
        # We are configuring agent to expect
        # to see same number as state
        assert self.state == action
        assert self.done == False
        self.state += 1
        if self.state == self.max_steps:
            self.done = True
        self.total_steps += 1
        # print("STEP", self.state, 1, self.done)
        WorkflowTestConstants.do_random_sleep()
        return self.state, 1, self.done, {}

    @record.event
    def reset(self):
        """
        This resets the environment.  It resets the internal state
        and the done flag.  We also count the total number of resets.

        Returns
        -------
        int
            The fresh restated state
        """
        self.state = 0
        self.done = False
        self.total_resets += 1
        return self.state

class FakeAgent(ExaAgent):
    """
    This is a fake agent.  It is used in coordination with the fake env
    to test a given workflow.  In this agent we have several counters
    (_has_data, _train, _target_train, _action, and _total_action) which
    we can query after running to ensure that the correct number of actions
    is taken.  We also have counters (_weights, _indices, and _loss) which
    always count up and are used to ensure correct coordination of the
    workflow.  These counters' values are asserted at runtime
    (i.e. workflow.run).  The previous counters are (mostly) asserted post
    run.

    The _weights counter is a list of two.  The first element is the counter
    for the learner.  The second is the counter for the actor.  This counter
    is split into two to support single rank tests.  The _indices counter is
    also similarly split.  For this counter the actor increments the second
    element and the learner will acknowledge its acceptance by
    setting the first element to the second element.  The third element of
    _indices is set to the actors rank for roundtrip verification.

    As workflows become uncoupled, it is harder to guarantee how often a
    model will be updated and how far off-policy learning can occurs.  We
    attempt to provide hooks for this using the WorkloadTestConstants.
    However, this will probably require tuning with a given workflow in
    conjunction with these tests!!!  This process should help the developer
    to understand how a workflow impacts the learning process.

    Attributes
    ----------
    name : string
        Name of the fake agent
    _has_data : int
        Counts how many times remember has been called
    _train : int
        Counts how many times train has been called
    _target_train : int
        Counts how many times target_train has been called
    _action : int
        Keeps track of the action to perform.  It counts up
        until the max number of steps for the workflow or
        environment is reached.
    _total_action : int
        Counts how many times action has been called
    _weights : List
        This is the fake weights to pass around.  The list contains
        two counter that counts up.  _weights[0] increase after a train
        performed by learner.  _weights[1] is set to _weights[0] when
        it is received in set_weights.
    _indices : List
        This is fake indices to pass around.  This is similar to
        the fake weights.  _indices[0] is incremented by
        generate data.  _indices[1] is set by the learner in train.
        _indices[2] is set to the global rank id.
    _weight_loss_check : List
        This list consists of the model indices of an actor
        (given by _weights[1]) when priority_replay is set.
        We use this list to check set priority is returning
        to the actor a valid set of "indices and loss"
    _state : int
        The expected state to see coming from the environment.
        This number increase by one unless it is reset when
        done is sent to remember.
    _observed_action : int
        This is the expected observed action coming from the
        environment.  We expect the action to increase by one
        after each call to step.
    _reward : int
        The expected reward from a step.  Is always set to one.
    _next_state : int
        The expected state from calling step.  This number should
        always be one larger than _state.
    _done : bool
        The expected done flag.  This should only change when
        the max number of steps for the environment or workflow
        is reached.
    _update_weights_on_get : bool
        This flag indicates when to increase the weights.  We
        update the weights on a get after a train.
    env : ExaEnv
        The fake environment
    is_learner : bool
        Flag indicating if we are a learner.
    priority_scale : float
        Indicates if we are using experience replay
    batch_size : int
        The max size of a single batch
    buffer_capacity : int
        The capacity of the memory for the agent
    epsilon : float
        Not well used...
    tau : float
        Commonly used in target train, but not in the fake one.
    _off_policy : List
        This list is the differences between a new model and
        old model recorded when setting weights by the actor
    _behind : List
        This list is the differences between a current model and
        the model "used to generate data" recorded during train
        by the learner.
    _priority_delay : List
        This list is the delay in models from the time between
        generate_data and set_priority observed in set_priority
        by an actor.
    """

    name = "FakeAgent-v0"

    def __init__(self, env=None, is_learner=False):
        """
        Parameters
        ----------
        env : ExaEnv
            The fake environment used in test
        is_learner : bool
            Indicates if agent is learner
        """
        # These are counters
        self._has_data = 0
        self._train = 0
        self._target_train = 0
        self._action = 0
        self._total_action = 0

        # Counter/Messages
        self._weights = [0, 0]
        self._indices = [0, 0, ExaComm.global_comm.rank]
        self._weights_loss_check = []

        # Expected values for remember
        self._state = 0
        self._observed_action = 0
        self._reward = 1
        self._next_state = 1
        self._done = False

        # We start this flag true for learner
        # so set_weights on actors see +1
        self._update_weights_on_get = is_learner

        # These are required members
        self.env = env
        self.is_learner = is_learner
        self.priority_scale = 1
        self.batch_size = 0
        self.buffer_capacity = 0
        self.epsilon = 1
        self.tau = 1

        # For post processing
        self._off_policy = []
        self._behind = []
        self._priority_delay = []

    @record.event
    def get_weights(self):
        """
        This returns the weights.  _weight[0] is updated when the
        train flag is set (meaning a target train has happened).
        We use this delay because set_weights expect the weights

        to be increased by at least one.  From the perspective of
        the learner this should only happen when train/target_train
        are called. On startup however we do a get and set without
        calling any trains.  By initializing the
        _update_weights_on_get we can
        have increase the weights on the first get.

        This also has the effect that multiple calls to train without
        calling get_weights will look like a single model update.
        This works with our approximation of on-policy learning.

        Returns
        -------
        list
            Weights to be sent
        """
        if self._update_weights_on_get:
            self._weights[0] += 1
            self._update_weights_on_get = False
        return self._weights[:]

    @record.event
    def set_weights(self, weights):
        """
        This sets the weights.  The workflow should call set_weights
        when it receives an update from the learner.  The weights
        for this fake agent are a constantly increasing counter(s) that
        keeps track of how often train/target_train have been called.
        From the point of view of the agents, we expect this value to
        always be increase by some amount.  For on-policy learning this
        should only increase by 1.  Off-policy learning will increase by
        some factor.  We can assert how far "off-policy" we find
        acceptable.

        Parameters
        ----------
        weights : list
            The fake weights (counter) to evaluate
        """
        # weights[0] is from the learner _weights[1] is on the actor
        assert weights[0] - self._weights[1] > 0, "set_weights: recv duplicate weights " + str(weights[0]) + ", " + str(self._weights[1])
        if WorkflowTestConstants.on_policy > -1:
            # This wont work for multiple agents... i.e. off-policy
            assert weights[0] - self._weights[1] <= WorkflowTestConstants.on_policy, ("set_weights: off-policy " + str(weights[0]) +
                                                                                      ", " + str(self._weights[1]) + ", policy " +
                                                                                      str(WorkflowTestConstants.on_policy))
        else:
            self._off_policy.append(weights[0] - self._weights[1])
        self._weights[1] = weights[0]

    @record.event
    def remember(self, state, action, reward, next_state, done):
        """
        This function is used to evaluate if the agent is observing the correct actions
        across the environment, agent, and workflow.  Each input has an expected state
        which is asserted.  If the max steps per episode given by the environment or
        workflow is given, the expected observations are reset.  This test should catch
        if the workflow is not updating the current_state = next_state.  This will
        also increse the _has_data counter which is used by generate data.  _has_data
        will indicate the total number of times remember has been called.

        Parameters
        ----------
        state : int
            This is the current state given by the workflow.
            The state should be increase by one each time except on reset.
            After a reset the value should be zero.
        action : int
            The is the action given by the workflow.  It should be
            a number increasing by one each time except on reset.
            After a reset the value should be zero.
        reward : int
            The reward given by the workflow.  This should be 1.
        next_state : int
            The next state following an action given by the workflow.
            This value should be one larger than the current state.
        done : bool
            This flag indicates if the environment is finished coming from
            the workflow.  This flag should be set if the max number of steps
            per episode for the environment or workflow has been reached.
        """
        assert self._state == state, "remember: state " + str(self._state) + " == " + str(state)
        self._state += 1
        assert self._observed_action == action, "remember: action " + str(self._observed_action) + " == " + str(action)
        self._observed_action += 1
        assert self._reward == reward, "remember: reward " + str(self._reward) + " == " + str(reward)
        assert self._next_state == next_state, "remember: next_state" + str(self._next_state) + " == " + str(next_state)
        self._next_state += 1

        # We only are concerned with done being set on env max not workflow max...
        if next_state == self.env.max_steps or next_state == WorkflowTestConstants.workflow_max_steps:
            assert done == (next_state == self.env.max_steps), "remember: done " + str(done) + " == " + str(next_state == self.env.max_steps)
            self._action = 0
            self._observed_action = 0
            self._state = 0
            self._next_state = 1
        else:
            assert done == False, "remember: done " + str(done) + " == " + str(False)

        # This counter store how many experience we have seen
        self._has_data += 1

    @record.event
    def train(self, batch):
        """
        This function is used to check if the states given by the environment are
        correct.  The generage_data function will pass back the weight counters
        given to it and a counter for indices/loss.  For on-policy learning
        the weights should be the same as the current weights.  For off-policy
        learning the weights should be some reasonable value less.  We use this
        function to keep a count of how many times train is called.

        We send back "indices and loss" with updated indices[0] to show we have
        processed the data. The remaining parts are untouched.

        Parameters
        ----------
        batch : pair
            The first value is the weights that the learner passed to the agent.
            The second value is indices counter given by the agent.

        Returns
        -------
        pair
            When using priority replay, the indices and weights are passed back
            for the agent to validate.
        """
        # For a real agent this would be:
        # states, target = batch
        weights, indices = batch
        assert weights[1] <= self._weights[0], "train: data from the future " + str(weights[0]) + ", " + str(self._weights[0])
        if WorkflowTestConstants.behind > -1:
            # This assert wont work for multiple agents
            assert self._weights[0] - weights[1] <= WorkflowTestConstants.behind, ("train: data is too old " + str(weights[0]) +
                                                                                   ", " + str(self._weights[0]) + ", behind " +
                                                                                   str(WorkflowTestConstants.behind))
        else:
            self._behind.append(self._weights[0] - weights[1])
        self._train += 1
        WorkflowTestConstants.do_random_sleep()

        if self.priority_scale:
            indices[0] = indices[1]
            return indices, weights

    @record.event
    def target_train(self):
        """
        Target train is called to "update the target model."  In reality
        this call is probably not need for anything not the sync workflow.
        Target train should be called after train.  We assert counter to
        make sure they are called in sync.
        """
        self._update_weights_on_get = True
        self._target_train += 1
        assert self._target_train == self._train, "target_train: " + str(self._target_train) + " == " + str(self._train)

    @record.event
    def action(self, state):
        """
        This is where an agent does inference for a given state to determine the
        next action.  In this function, we increase the action at the same rate
        as the state.  We assert the workflow is giving us the correct state.
        The action is reset in the remember function when we know done has been
        observed by the workflow.  We also count the total number of actions
        performed.

        Parameters
        ----------
        state : int
            Current state given by the workflow.  This should match what we our
            expected state in self._state.

        Returns
        -------
        int
            The action counter
        """
        assert self._state == state, "action: " + str(self._state) + " == " + str(state)
        assert self._action < self.env.max_steps, "action: " + str(self._action) + " < " + str(self.env.max_steps)
        ret = self._action
        self._action += 1
        self._total_action += 1
        return ret, 1

    @record.event
    def has_data(self):
        """
        This returns if remember has been called yet (i.e. _has_data counter).
        This should correspond to running a step and storing the result in the agent.

        Returns
        -------
        bool
            If remember has been called.
        """
        return self._has_data > 0

    @record.event
    def generate_data(self):
        """
        This is a generator that returns the current weights of an agent.  This
        is linked to the train function of the learner who is expecting to
        receive the weights counter back.  We increment the indices counter to
        keep track of how many times generate data is called before that batch has
        been processed.

        Returns
        -------
        pair
            The weights counters to send to learner's train function.
        """
        self._indices[1] += 1
        self._weights_loss_check.append(self._weights[1])
        # This assert might fail with RMA since it might
        # "get" the same model multiple times
        assert len(self._weights_loss_check) == len(set(self._weights_loss_check)), "There are multiple batches generated from a single model"
        yield self._weights[:], self._indices[:]

    @record.event
    def set_priorities(self, indices, loss):
        """
        This function takes the indices and loss counters sent from the learner.
        In this case the indices are a counter and the loss is the weights counter
        sent from the agent to the learner and back to the agent for round-trip
        verification. We expect them to be the same as what we sent.  Since then
        the workflow may have updated our weights, sent another batch, or both,
        the difference between weights and indices counter will let us know how
        long it is taking for the round trip of set_priorities.

        Parameters
        ----------
        indices : list
            Counter originally coming from agent, sent to learner, and sent back to agent
        loss : list
            Counter of the weights used when training.  See comment in function.
        """
        weights = loss
        assert indices[0] == self._indices[1], "set_priorities: " + str(indices[0]) + " == " + str(self._indices[1])
        assert indices[2] == ExaComm.global_comm.rank, "indices are not for this rank " + str(ExaComm.global_comm.rank) + " " + str(indices[2])
        assert weights[1] in self._weights_loss_check, "set_priorities: " + str(weights[1]) + " not in " + str(self._weights_loss_check)
        self._weights_loss_check.remove(weights[1])
        self._priority_delay.append(self._weights[1] - weights[1])


if __name__ == "__main__":
    """
    This is if we want to run this outside of pytest.
    """
    # Defaults
    num_learners = 1
    procs_per_env = 1
    workflow_name = 'sync'
    episodes = 10
    steps = 10

    # Command line parameters
    if len(sys.argv) == 6:
        num_learners = int(sys.argv[1])
        procs_per_env = int(sys.argv[2])
        workflow_name = str(sys.argv[3])
        episodes = int(sys.argv[4])
        steps = int(sys.argv[5])
    else:
        print("[STAND-ALONE WORKFLOW TEST] Require 5 args: num_learners procs_per_env workflow_name episodes steps")
        print("[STAND-ALONE WORKFLOW TEST] Running with default.")

    print("[STAND-ALONE WORKFLOW TEST] num_learners:", num_learners,
          "procs_per_env", procs_per_env,
          "workflow_name", workflow_name,
          "episodes", episodes,
          "steps", steps,
          flush=True)

    # Set constants
    WorkflowTestConstants.episodes = episodes
    WorkflowTestConstants.env_max_steps = steps
    WorkflowTestConstants.workflow_max_steps = steps

    # Set params
    dir_name = './log_dir'
    initialize_parameters(params={"mpi4py_rc": "false",
                                  "log_level": [3, 3],
                                  "output_dir": dir_name,
                                  "episode_block": "false",
                                  "batch_frequency": 1,
                                  "n_episodes": episodes,
                                  "n_steps": steps,
                                  "save_weights_per_episode": "false",
                                  "profile": "None"})

    # Set up comm
    ExaSimple(None, procs_per_env, num_learners)

    # Make log dir
    rank = ExaComm.global_comm.rank
    made_dir = False
    if rank == 0 and not os.path.isdir(dir_name):
        os.mkdir(dir_name)
        made_dir = True

    # Register fake env and agent
    gym.envs.registration.register(id=FakeEnv.name, entry_point=FakeEnv)
    exarl.agents.registration.register(id=FakeAgent.name, entry_point=FakeAgent)

    # Create fake env, agent, and learner with real workflow
    env = None
    agent = None
    if ExaComm.is_actor():
        env = ExaEnv(gym.make(FakeEnv.name).unwrapped)
    if ExaComm.is_agent():
        agent = exarl.agents.make(FakeAgent.name, env=env, is_learner=ExaComm.is_learner())
    workflow = exarl.workflows.make(workflow_name)
    learner = FakeLearner(episodes, steps, agent, env, workflow, dir_name)

    learner.run()
    learner.print_delays()

    # Run and print record if failure
    # try:
    #     print("[STAND-ALONE WORKFLOW TEST] Running from rank", ExaComm.global_comm.rank, flush=True)
    #     learner.run()
    # except Exception as e:
    #     print(e)
    #     for i in record.events:
    #         print(i)

    # Clean up log if created
    ExaComm.global_comm.barrier()
    if made_dir:
        os.rmdir(dir_name)
