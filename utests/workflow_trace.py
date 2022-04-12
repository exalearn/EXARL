import sys
import os
import numpy as np
import gym
from gym import spaces
import exarl
from exarl.utils import candleDriver
from exarl.base.comm_base import ExaComm
from exarl.network.simple_comm import ExaSimple
from exarl.base.env_base import ExaEnv
from exarl.base.agent_base import ExaAgent
from exarl.envs.env_vault.UnitEvn import EnvGenerator
import functools
import time
import random

class record:
    """"
    This class implements vector clocks:
    https://en.wikipedia.org/wiki/Vector_clock
    https://people.cs.rutgers.edu/~pxk/417/notes/logical-clocks.html
    Vector clock give us the ability to create partial orderings.
    We can use this to under stand what things are happening in parallel
    and this gives us a better view of how workflows are working.

    Vector clocks are implemented by having p counters (i.e. clock) on
    each process (i.e. rank).  P is the total number of processes (thus
    we have P^2 total clock across all ranks).  We increase our clock
    (i.e. clock[rank]) every event we encounter.  When we "send" a
    receive a message we update our vector of clocks to the max time
    (i.e. count) observed by each clock.

    When an event occurs and we increase our clock, we also record
    the event with a timestamp (i.e. the state of a rank's vector clock).
    At the end we can create a partial order by merging the records
    from every rank into one based on two key conditions:

    1. Two event happen at the same time if all of the clock are equal.
    2. Assuming two event have clocks that are not equal, an event a
    happens before event b if all of a's clocks are less than or equal
    to b's clocks

    To check for greater than we invert condition 2 and evaluate b vs a.
    If the events are neither = < >, then the events are happening
    "concurrently" (i.e. we have a partial order).

    This is not a complete DAG, as we are still developing the partial
    order based on the observed order of events.  In otherwords, events
    could be reordered from the point of view of execution reduce the
    number of dependencies.

    To create a complete view of the events across a run, we gather all
    the events from each node and stitch them together.  Since events
    are recorded in order on a single node, we do not need to perform
    a complete sort (rather we have just log(n) merging).

    Events are recorded via a decorator, which we attach to each function
    we care about.  Functions that can be seen as message sends, will
    call get_clock_to_send while messages that receive will call update_clocks.

    Attributes
    ----------
    counters : Dictionary
        This contains a name of each function and how many times it ran
    events : List
        This contains a list of events that have occurred on a single node
    clocks : list
        These are clocks, one for each rank
    """
    counters = None
    events = None
    clocks = None

    def reset(num_nodes):
        """
        Resets the record.

        Parameters
        ----------
        verbose : bool
            Flag indicating to print on each event
        """
        record.counters = {}
        record.events = []
        record.clocks = [0] * num_nodes

    def get_clock_to_send():
        """
        This function adds one to ranks clock and returns
        a copy of the clock to send.

        Returns
        -------
        list
            A copy of the vector clock
        """
        record.clocks[ExaComm.global_comm.rank] += 1
        return record.clocks[:]

    def update_clock(clocks):
        """
        This receives a vector clock and updates the local
        vector clock with the max per rank.

        Parameters
        ----------
        clocks : list
            Incoming vector clock
        """
        record.clocks = [max(a, b) for a, b in zip(record.clocks, clocks)]

    def equal(A, B):
        """
        This compares two clocks to see if they are equal.
        They are only equal if all of clocks are equal.

        Parameters
        ----------
        A : list
            Vector clock
        A : list
            Vector clock
        """
        for a, b in zip(A[2], B[2]):
            if(a != b):
                return False
        return True

    def less_than(A, B):
        """
        This compares two clocks to see if A is less than B.
        We assume that A != B !!!  A's clocks must be
        less than or equal to B for event A to have occurred
        before B.

        Parameters
        ----------
        A : list
            Vector clock
        B : list
            Vector clock

        Returns
        -------
        bool
            If A occurred before B
        """
        # We assume that A != B
        for a, b in zip(A[2], B[2]):
            if(a > b):
                return False
        return True

    def compare(A, B):
        """
        This does a full comparison of two vector clocks.  We return:
        -1 iff A occurs before B
         1 iff B occurs before A
         0 otherwise
        We do this by checking A == B, A < B, B < A in order.

        Parameters
        ----------
        A : list
            Vector clock
        B : list
            Vector clock

        Returns
        -------
        int
            -1 iff A occurs before B, 1 iff B occurs before A, and 0 otherwise
        """
        if record.equal(A, B):
            return 0
        if record.less_than(A, B):
            return -1
        # We still have to check again because the
        # two could still be concurrent
        if record.less_than(B, A):
            return 1
        # They are concurrent!
        return 0

    def sort(data):
        """
        This function sorts events from every rank.
        We assume that events within a rank are ordered
        based on observation.

        Parameters
        ----------
        data : list
            List of list of events

        Returns
        -------
        list
            Single list of all events
        """
        total_size = sum([len(i) for i in data])
        total = data[0]
        for B in data[1:]:
            aIndex = 0
            bIndex = 0
            while bIndex < len(B):
                if aIndex == len(total):
                    total += B[bIndex:]
                    break
                elif record.compare(total[aIndex], B[bIndex]) == 1:
                    total.insert(aIndex, B[bIndex])
                    bIndex += 1
                aIndex += 1
        assert total_size == len(total), str(total_size) + " vs " + str(len(total))
        return total

    def event(func):
        """
        This is a decorator to put ontop of a function.  It will record
        how many times it has been call in the counters dictionary and
        add an entry under events.  It will also increment the appropriate
        clock within the vector clocks.

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
            result = func(*args, **kwargs)
            if func.__name__ in record.counters:
                record.counters[func.__name__] += 1
            else:
                record.counters[func.__name__] = 1
            record.clocks[ExaComm.global_comm.rank] += 1
            record.events.append((ExaComm.global_comm.rank, record.counters[func.__name__], record.clocks[:], func.__name__,))

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
    rank_sleep = False
    random_sleep = False

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
        # Doubt we accually need self.global_comm and self.global_size
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
    state : int
        The current state of the environment.  Is int because we are using discrete space.
    max_steps : int
        The max steps the environment can take.
    """

    name = "FakeEnv-v0"

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(WorkflowTestConstants.env_max_steps)
        self.observation_space = spaces.Discrete(WorkflowTestConstants.env_max_steps)

        self.state = 0
        self.max_steps = WorkflowTestConstants.env_max_steps

    @record.event
    def step(self, action):
        """
        This is a single step.  If number of calls == max steps
        done is set.  State is always incremented by 1.

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
        if self.state < self.max_steps:
            self.state += 1
        done = self.state == self.max_steps

        WorkflowTestConstants.do_random_sleep()
        return self.state, 1, done, {}

    @record.event
    def reset(self):
        """
        This resets the environment.

        Returns
        -------
        int
            The fresh restated state
        """
        self.state = 0
        return self.state

class FakeAgent(ExaAgent):
    """
    This class is a fake agent.  Each method is tagged with vector clocks to
    keep track of events.  A send is a method that will generate data for
    another rank.  A receive is a method that will take data in from another
    rank.  Instead of passing weight, indices, or loss around, we replace
    this with passing the vector clocks allowing us to create the partial
    orders.

    Attributes
    ----------
    name : string
        Name of the fake agent
    _has_data : int
        Counts how many times remember has been called
    _data : int
        Garbage data to satisfy the api requirements
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
        self._has_data = 0
        self._data = [0]

        # These are "required" members
        self.env = env
        self.is_learner = is_learner
        self.priority_scale = 1
        self.batch_size = 0
        self.buffer_capacity = 0
        self.epsilon = 1
        self.tau = 1

    @record.event
    def get_weights(self):
        """
        Getting the weights can be seen as a send
        function as the weights are collected to pass
        to another rank.  We replace the weights with
        the clock.

        Returns
        -------
        list
            Weights aka the rank's vector clock
        """
        # This is where the learner sends to the actors
        return record.get_clock_to_send()

    @record.event
    def set_weights(self, weights):
        """
        This is the receive for the vector clocks originating
        from get_weights.  We update our local clock with
        incoming clocks.

        Parameters
        ----------
        weights : list
            This is the incoming vector clock
        """
        # This is where actor receives learner
        record.update_clock(weights)

    @record.event
    def remember(self, state, action, reward, next_state, done):
        """
        We keep track that data has been generated, but do not
        require to keep track of anything else.  All parameters
        are ignored.  When using priority_scale this function
        sends its vector clock back to the agent.
        """
        self._has_data += 1

    @record.event
    def train(self, batch):
        """
        This is the receive of a vector clock coming from a agent rank.
        We update our clock accordingly.

        Parameters
        ----------
        batch : pair
            The first is junk data.  The second is the incoming vector clock.

        Returns
        -------
        pair
            When using priority replay, the first value is junk and the second
            is our vector clock.
        """
        _, clocks = batch
        # This is where the learner receives from the agent
        record.update_clock(clocks)
        WorkflowTestConstants.do_random_sleep()
        if self.priority_scale:
            # This is where the learner sends to the agent
            return self._data, record.get_clock_to_send()

    @record.event
    def target_train(self):
        """
        This function does not need to do anything
        but record via its decorator.
        """
        pass

    @record.event
    def action(self, state):
        """
        This function is a send from the lead of the environment
        to the rest of the environment ranks in its env comm.

        Parameters
        ----------
        state : int
            Don't care

        Returns
        -------
        List
            Our vector clock to send
        """
        return record.get_clock_to_send(), 1

    @record.event
    def has_data(self):
        """
        This returns if remember has been called yet (i.e. _has_data counter).
        This should correspond to the total ran steps for a given agent.

        Returns
        -------
        bool
            If remember has been called.
        """
        return self._has_data > 0

    @record.event
    def generate_data(self):
        """
        This is generator is a send from the agent to the learner.

        Returns
        -------
        pair
            The first value is junk and the second is our vector clock.
        """
        # This is where the agent sends to the learner
        yield [self._data], record.get_clock_to_send()

    @record.event
    def set_priorities(self, indices, loss):
        """
        This is the receive function coming from the learners train
        when priority_scale is turned on.

        Parameters
        ----------
        indices : list
            Don't care
        loss : list
            This is the incoming vector clock
        """
        record.update_clock(loss)

if __name__ == "__main__":
    """
    This is if we want to run this outside of pytest.
    """
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

    # Init comms and record
    ExaSimple(procs_per_env=procs_per_env, num_learners=num_learners)
    record.reset(ExaComm.global_comm.size)

    # Make log dir
    rank = ExaComm.global_comm.rank
    made_dir = False
    dir_name = './log_dir'
    if rank == 0 and not os.path.isdir(dir_name):
        os.mkdir(dir_name)
        made_dir = True

    candleDriver.run_params = {'output_dir': dir_name}

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

    # Run workflow
    learner.run()
    # record.print()

    # Collect records, sort, and print
    if ExaComm.global_comm.rank:
        ExaComm.global_comm.send(record.events, 0)
    else:
        data = None
        all = [record.events]
        for i in range(1, ExaComm.global_comm.size):
            data = ExaComm.global_comm.recv(data, source=i)
            all.append(data)
        all = record.sort(all)

        last = 0
        group = [(0, 0)]
        for i in range(1, len(all)):
            res = record.compare(all[i - 1], all[i])
            if res == -1:
                last += 1
            group.append((last, res))

        for i, j in zip(all, group):
            print(i[0], i[1], j[0], j[1], i[2], i[3])

# TODO: Make test out of record
# Use batch size for the lower bound on over-training
# Use memory size / window size for losing data

# To figure out how many look at sorted list and find where comp = 0
# then back track to where learner
