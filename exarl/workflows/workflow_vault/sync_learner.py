# This material was prepared as an account of work sponsored by an agency of the
# United States Government.  Neither the United States Government nor the United
# States Department of Energy, nor Battelle, nor any of their employees, nor any
# jurisdiction or organization that has cooperated in the development of these
# materials, makes any warranty, express or implied, or assumes any legal
# liability or responsibility for the accuracy, completeness, or usefulness or
# any information, apparatus, product, software, or process disclosed, or
# represents that its use would not infringe privately owned rights. Reference
# herein to any specific commercial product, process, or service by trade name,
# trademark, manufacturer, or otherwise does not necessarily constitute or imply
# its endorsement, recommendation, or favoring by the United States Government
# or any agency thereof, or Battelle Memorial Institute. The views and opinions
# of authors expressed herein do not necessarily state or reflect those of the
# United States Government or any agency thereof.
#                 PACIFIC NORTHWEST NATIONAL LABORATORY
#                            operated by
#                             BATTELLE
#                             for the
#                   UNITED STATES DEPARTMENT OF ENERGY
#                    under Contract DE-AC05-76RL01830
import time
import csv
import numpy as np
import exarl
from exarl.utils.globals import ExaGlobals
from exarl.base.comm_base import ExaComm
from exarl.network.typing import TypeUtils
from exarl.utils.profile import PROFILE
from exarl.utils.introspect import introspect

class SYNC(exarl.ExaWorkflow):
    """
    This class implements a workflow by breaking the functionality into pieces.
    We define 3 key terms used throughout the implementation/documentation:

    Learner - the rank responsible for running the train/target train functions.
    Agents - the ranks responsible for training/inference.  Agents include learners.
    Actors - everyone that is not a learner.

    In addition to this there is the environment comm which includes an agent and
    actors.

    The following is an example useful for understanding the above descriptions.
    The example assumes 14 ranks (2 Learners, 5 Agents, 4 Environment)
    We present both the comms and the ranks.
    There is no actor comm so we add * to depict actors.
    Rank   0  1  2  3  4  5  6  7  8  9  10 11 12 13 14
    Learn     0  1  -  -  -  -  -  -  -  -  -  -  -  -
    Agent     0  1  2  -  -  -  3  -  -  -  4  -  -  -
    Actor     -  -  *  *  *  *  *  *  *  *  *  *  *  *
    Envir     -  -  0  1  2  3  0  1  2  3  0  1  2  3

    For a single rank execution, global rank 0 is the learner, agent, and environment.

    There are two possible cut-off conditions that this workflow respects:

    1. The first cut-off condition for this is based on how many completed (done)
    episodes the learner observes.

    2. The second is based on learning convergence.  This is configured via the config
    files using the rolling reward length and cutoff settings.  We determine if we
    have converged by looking at the rolling average of the absolute value of the
    differences across the last N number of episodes.  If this value is <= the config
    cutoff value, we terminate execution.  To turn the cutoff off
    set the cutoff configuration to -1.

    We have two sets of internal variables one set used by the learners and another
    used by the actors. Batches and weights are passed via the leaner and actor calls
    by setting self.batch and self.weights.

    We maintain a distinction between learner and agent variables even in the single
    rank such that this can serve as a base class for future workflows.  The actor
    and learner functions should remain unchanged.  The easiest changes should be
    made to the send/recv, init_learner, and run functions.

    TODO:
        - Add multi-learner - We can think if this should be in this class or in
        a inherited class.

    Attributes
    ----------
    block_size : int
        This is used to indicate we want on-policy learning
        and is used by the learner. When we want on-policy
        learning we set this to the total number of agents.
        We can allow off-policy learning by setting
        the value to one.  For sequential learning this
        is set to one.

    batch_step_frequency : int
        This value is used to determine how often we should
        send a batch of data within an episode.  The value
        represents performing batch_step_frequency steps
        per 1 batch send.

    batch_episode_frequency : int
        This indicates how many episodes an actor should run before sending
        its results to the learner.  This features is required by sum learning
        algorithms (i.e. rollout).

    log_frequency : int
        This indicates how often we should write to the log.  Logging
        can be costly in time.

    clip_rewards : list
        This variable indicates if rewards should be clipped to a space
        of -1 to 1.

    next_episode : int
        This value is used by the learner to keep track of
        what the next episode to run is.

    done_episode : int
        This is a counter of how many episodes have been
        finished used by the learner.

    episode_per_rank : list
        This is a list of ints that indicate what episode
        each rank is working on. This is used by the learner.

    total_reward : int
        This is the sum of rewards for the current episode.
        This is used by an actor.

    steps : int
        This is the current step within an episode.
        This is used by the actor

    done : bool
        This flag indicates if the episode is done.
        This is used by the actor.

    current_state :
        This is the current state of an environment for
        a given actor.

    model_count : int
        This is the current generation of the model.  This counter
        is set on the learner.  It is up to the send/recv to
        propagate this to the actors.

    save_weight_per_episode : bool
        This will cause the workflow to save the weights for each
        learner model generation.

    filename_prefix : string
        Prefix of the log file.

    train_file : file
        File we use to write logs to.

    train_writer : csv.writer
        Logger that writes to train_file.

    episode_reward_list : list
        This store total reward at the end of an episode.

    cutoff : float
        This is the minimum value of absolute difference between the
        last rolling_reward_length episodes required to consider
        learning to have converged.  Set to -1 to turn off the
        convergence cutoff.

    rolling_reward_length :
        The last number of episodes to consider to a
        rolling average reward

    converged : bool
        Indicates if learning convergence has been reached

    alive :
        Counter of the number of actors that have not finished.

    verbose : bool
        Debug print flag


    """
    verbose = False

    def __init__(self, agent=None, env=None):
        self.debug('Creating SYNC', ExaComm.global_comm.rank, ExaComm.is_learner(), ExaComm.is_agent(), ExaComm.is_actor())

        self.block_size = 1
        block = TypeUtils.get_bool(ExaGlobals.lookup_params('episode_block'))
        if block:
            if ExaComm.global_comm.rank == 0:
                self.block_size = ExaComm.agent_comm.size
            self.block_size = ExaComm.global_comm.bcast(self.block_size, 0)

        # How often do we send batches
        self.batch_episode_frequency = ExaGlobals.lookup_params('batch_episode_frequency')
        if self.batch_episode_frequency <= 1:
            self.batch_step_frequency = ExaGlobals.lookup_params('batch_step_frequency')
            # Handles if < 1 was passed.  Must be at least 1
            self.batch_episode_frequency = 1
        else:
            # This is for multi-episode agents.  We will set the batch_step_frequency
            # to -1 since we only want to send full episodes.
            self.batch_step_frequency = -1

        # If it is set to -1 then we only send an update when the episode is over
        if self.batch_step_frequency == -1:
            self.batch_step_frequency = ExaGlobals.lookup_params('n_steps')

        # How often to write logs (in episodes)
        self.log_frequency = ExaGlobals.lookup_params('log_frequency')
        # If it is set to -1 then we only log at the end of the run
        if self.log_frequency == -1:
            self.log_frequency = ExaGlobals.lookup_params('n_episodes')

        self.clip_rewards = ExaGlobals.lookup_params('clip_rewards')
        if not self.clip_rewards:
            self.clip_rewards = None
        elif self.clip_rewards == True:
            self.clip_rewards = [-1, 1]

        # Learner episode counters
        self.next_episode = 0
        self.done_episode = 0
        self.episode_per_rank = None
        self.train_return = None
        if ExaComm.is_learner():
            self.episode_per_rank = [0] * ExaComm.agent_comm.size
            self.train_return = [None] * ExaComm.agent_comm.size

        # Actor counters
        self.total_reward = 0
        self.steps = 0
        self.done = True
        self.current_state = None

        # Learner counters
        self.model_count = 0

        # Initialize logging
        self.init_logging()

        # Save weights after each episode
        self.save_weights_per_episode = TypeUtils.get_bool(ExaGlobals.lookup_params('save_weights_per_episode'))

        # Check this for convergence
        self.episode_reward_list = []
        self.cutoff = ExaGlobals.lookup_params('cutoff')
        self.rolling_reward_length = ExaGlobals.lookup_params('rolling_reward_length')
        self.converged = False
        self.alive = 0

    def debug(self, *args):
        """
        Function to turn on and off debug print statements
        """
        if SYNC.verbose:
            print("[", self.__class__.__name__, ExaComm.global_comm.rank, "]", *args, flush=True)

    def init_logging(self):
        """
        Initialize the logging on rank 0.
        """
        # Get parameters
        self.results_dir = ExaGlobals.lookup_params('output_dir')
        self.nepisodes = ExaGlobals.lookup_params('n_episodes')
        self.nsteps = ExaGlobals.lookup_params('n_steps')

        # Do the initialization
        if ExaComm.is_agent():
            self.filename_prefix = 'ExaLearner_Episodes%s_Steps%s_Rank%s_memory_v1' % (str(self.nepisodes), str(self.nsteps), str(ExaComm.agent_comm.rank))
            self.train_file = open(self.results_dir + '/' + self.filename_prefix + ".log", 'w')
            self.train_writer = csv.writer(self.train_file, delimiter=" ")
            self.data_matrix = []

    def write_log(self, current_state, action, reward, next_state, total_reward, done, episode, steps, policy_type):
        """
        Rank zero writes the input data to the log file.

        Parameters
        ----------
        current_state : gym.space
            The state from the observation space at the current step

        action : gym.space
            The action from the action space given via inference

        reward : float
            The value of the state transition given by the environment
            performing a step

        next_state : gym.space
            The resulting state after performing action on current observation space

        total_reward : float
            This is the cumulative reward for within an episode

        done : bool
            Flag indicated the episode ended

        episode : int
            Current episode to log

        steps : int
            This is the current step within an episode

        policy_type : int
            This value is given by the action.
        """
        if ExaComm.is_agent():
            self.data_matrix.append([time.time(), current_state, action, reward, next_state, total_reward, done, episode, steps, policy_type])
            if done or self.converged:
                if (episode == (self.nepisodes - 1)) or ((episode + 1) % self.log_frequency == 0) or self.converged:
                    self.train_writer.writerows(self.data_matrix)
                    self.train_file.flush()
                    self.data_matrix = []

    def save_weights(self, exalearner, episode, nepisodes):
        """
        This function is a wrapper around save weights.  If save_weights_per_episode flag
        is set in configuration, we will store all the weights for each model generation.
        Otherwise, we just record the final weights.

        Parameters
        ----------
        exalearner : ExaLearner
            This contains the agent and env

        episode : int
            Current episode to index weights by

        nepisodes : int
            Total number of episodes to be performed
        """
        if self.save_weights_per_episode and episode != nepisodes:
            exalearner.agent.save(exalearner.results_dir + '/' + self.filename_prefix + '_' + str(episode) + '.h5')
        elif episode == nepisodes or self.converged:
            exalearner.agent.save(exalearner.results_dir + '/' + self.filename_prefix + '.h5')

    # TODO: What to do about dst?
    @introspect
    def send_model(self, exalearner, episode, train_return, dst):
        """
        This function is responsible for sending the model from the learner to
        other agents.  For the sync learner, we just store the weights in the workflow.
        For more interesting workflows, this should include an MPI send or RMA operation.
        We intend for this function is to be overloaded in subsequent workflows.

        The workflow expects a message containing the episode, and the model weights.
        To use the learner and actor functions, this must be respected.  Otherwise, it
        one will need to rewrite those functions in a derived class.

        Parameters
        ----------
        exalearner : ExaLearner
            This contains the agent and env

        episode : int
            The current episode corresponding to the model generation

        train_return : list
            This is what comes out of the learner calling train to be sent back
            to the actor (i.e. indices and losses).

        dst : int
            This is the destination rank given by the agent communicator
        """
        weights = exalearner.agent.get_weights()
        self.weights = [episode, weights, []]
        if train_return:
            self.weights[-1].append(train_return)

    @introspect
    def recv_model(self):
        """
        This function is the corresponding receive function to
        the send_model function.  Here the weights are being received by the
        the other agents (sent from the learner).  For the sync learner
        we retrieve this data which is stored locally, however this function is
        to be overloaded for more interesting workflows.

        Returns
        -------
        list :
            This list should contain the episode, model weights,
            and the train return (indices and losses if turned on)
        """
        return self.weights

    @introspect
    def send_batch(self, batch_data, policy_type, done, episode_reward):
        """
        This function is used to send batches of data from the actor to the
        learner.  For the sync learner, data is being stored locally.  This
        function is intended to be overwritten by future workflows.

        Parameters
        ----------
        batch_data : list
            This is a list of experiences generate by the actor to send to
            the learner.

        policy_type : int
            This is the policy given by the actor performing inference to get an action

        done : bool
            Indicates if the episode is competed

        episode_reward : float
            The total reward from the last episode.  If the episode in not done, it
            will be the current total reward.
        """
        self.batch = [ExaComm.agent_comm.rank, batch_data, policy_type, done, episode_reward]

    @introspect
    def recv_batch(self):
        """
        This function is the corresponding receive function to the send_batch
        function.  Here the batch data is received by the learner
        (sent from an actor).  Again for the sync learner we retrieve this
        data which is stored locally, however this function is to be overloaded
        for more interesting workflows.

        Returns
        -------
        list :
            This list should contain the rank, batched data, policy type, and done flag.
            The done flag indicates if the episode the actor was working on finished.
        """
        return self.batch

    @introspect
    def reset_env(self, exalearner):
        """
        This function resets an environment if the done flag has been set.

        Parameters
        ----------
        exalearner : ExaLearner
            This contains the agent and env
        """
        if self.done:
            self.total_reward = 0
            self.steps = 0
            self.done = False
            self.current_state = exalearner.env.reset()
            print("RESET:", self.current_state, flush=True)

    def init_learner(self, exalearner):
        """
        This function is used to initialize the model on every agent.  The learner
        is responsible for sending out the model to each actor.  The actors will
        read in the values in the actor function.  The learner uses the
        episode_per_rank to keep track of which rank each episode is running and
        used next_episode to record which episode is next to send to an actor.

        We use episode_per_rank to store the current episode for when we batch
        the model updates ensuring on-policy learning.  We will determine we
        are finished by evaluating the done_episodes NOT episode_per_rank or
        next episode.

        For this sync learning we are assuming that the learner and actor
        are running on the same rank.  In this case we only care about
        episode_per_rank[0].  This function should be overwritten for the
        multi-agent case which should iterate over the agent comm and update
        the episode_per_rank[i] where i is the agent_comm.rank.

        The alive variable is used to know how many actors are still running.
        This is important when a cutoff or number of done episodes is reached.

        Parameters
        ----------
        exalearner : ExaLearner
            This contains the agent and env
        """
        if ExaComm.is_learner():
            # We are assuming there is only one right here
            self.episode_per_rank[0] = self.next_episode
            self.send_model(exalearner, self.next_episode, None, 0)
            self.next_episode += self.batch_episode_frequency
            self.alive += 1

    def check_convergence(self):
        """
        This function checks if our learning performance has converged.
        We consider this to converge if the past N episodes and an average
        absolute difference less than some configurable cutoff.  To use the
        convergence check, set the rolling_reward_length > 1 (config variable)
        and set the desired minimum via cutoff configuration variable.  To
        turn off set the cutoff to -1.

        Returns
        -------
        float :
            The average absolute difference if checking for convergence, -1 otherwise
        """
        # Lets us know how we are doing
        if self.cutoff > 0 and self.rolling_reward_length > 1 and not self.converged:
            if len(self.episode_reward_list) >= self.rolling_reward_length:
                ave = np.mean(np.abs(np.diff(np.array(self.episode_reward_list[-self.rolling_reward_length:]))))
                if ave < self.cutoff:
                    self.converged = True
                    print("Converged:", len(self.episode_reward_list), "Alive:", self.alive, "Ave:", ave, "Last:", self.episode_reward_list[-1])
                return ave
        return -1

    def inc_episode(self):
        """
        We abstract this increment for future workflows (RMA) which
        need to synchronize this value.

        Returns
        -------
        int :
            The next episode index
        """
        ret = self.next_episode
        self.next_episode += self.batch_episode_frequency
        return ret

    @introspect
    def learner(self, exalearner, nepisodes, start_rank):
        """
        This function is performed by the learner.  The learner
        performs the following key steps:

        1. Receives batches of experiences
        2. Trains the models on the data received
        3. Checks if an episode has finished
        4. Sends data back to the appropriate actors

        Each call to train/target train represents a new model
        generation.  On-policy learning means that the experiences
        contained by the batch are from the previous model (i.e.
        there is only one generation of models between them).
        Off-policy learning is done when experiences are used from
        previous model generations to train.  Training with a
        single actor will result in on-policy learning.  When we
        scale to use multiple actors we will by definition be
        training with older models.  If we were to collect data
        from each actor round robin with N actors, the model would
        be off policy by N models.  This assumes we collect the
        experiences from each actor and train one at a time.

        The block_size variable is to approximate on-policy learning
        with multiple actors.  By setting block_size to the number of
        actors, we will only send a new identical model to all actors
        after we have received and processed data from each actor.

        The start_rank is used to indicate the first rank actor rank
        in the agent comm.  For sync learner this is rank 0 since
        the learner and actor are on the same rank.  For others this
        should correlate to the number of learners.

        In summary:

        For single-actor:
            start_rank = 0
            block_size = 1

        For multi-actor
            start_rank = number of learners
            For blocking (on-policy)
                block_size = size of agent_comm
            For non-blocking (off-policy)
                block_size = number of learners + 1

        Ideally this function should not have to be overloaded.

        Parameters
        ----------
        exalearner : ExaLearner
            This contains the agent and env

        nepisodes : int
            The number of episodes to be performed

        start_rank : int
            The rank of the first actor
        """
        ret = False
        to_send = []
        # JS: The zip makes sure we have ranks alive
        for dst, _ in zip(range(start_rank, self.block_size), range(self.alive)):
            src, batch, policy_type, done, total_reward = self.recv_batch()
            self.train_return[src] = exalearner.agent.train(batch)

            self.model_count += 1
            to_send.append(src)

            if done:
                self.episode_reward_list.append(total_reward)
                self.done_episode += self.batch_episode_frequency
                self.episode_per_rank[src] = self.inc_episode()
                ret = True

            if self.converged:
                self.episode_per_rank[src] = nepisodes

        for dst in to_send:
            self.send_model(exalearner, self.episode_per_rank[dst], self.train_return[dst], dst)
            if self.episode_per_rank[dst] >= nepisodes:
                self.alive -= 1

        self.save_weights(exalearner, self.done_episode, nepisodes)
        return ret

    @introspect
    def actor(self, exalearner, nepisodes):
        """
        This function is performed by actors.  It performs the follow:

        1. Receives model weights from the learner
        2. Set agents with new model weights
        3. Resets the environment if necessary
        4. Performs inference based on current state to get action
        5. Broadcasts that action to the other ranks in the env_comm
        6. Performs the action and determines the reward and next state
        7. Records the experience (i.e. states, action, and reward)
        8. Check for max number of steps and broadcast
        9. Updates the current state to the new state
        10. Sends batches of experiences to the learner

        We use the batch_frequency to determine how often we send
        results back the learner.  To send data only on a complete
        episode, batch_frequency should be set to the max number
        of steps.

        Parameters
        ----------
        exalearner : ExaLearner
            This contains the agent and env

        nepisodes : int
            Total number of episodes to run

        Returns
        -------
        bool
            This function returns False if it receives an
            episode index >= the number of episodes to run.
            Otherwise it returns True.
        """
        # These are for ranks > 0
        episode = 0
        action = 0
        policy_type = 0

        # Get model and update the other env ranks (1)
        if ExaComm.env_comm.rank == 0:
            episode, weights, train_ret = self.recv_model()
        episode = ExaComm.env_comm.bcast(episode, 0)
        if episode >= nepisodes:
            return False

        # Set agent for rank 0 (2)
        if ExaComm.env_comm.rank == 0:
            exalearner.agent.set_weights(weights)
            for x in train_ret:
                exalearner.agent.train_return(x)

        # Repeat steps 3-9 for a number of episodes
        for eps in range(self.batch_episode_frequency):
            # Set the episode for envs that want to keep track
            exalearner.env.set_episode_count(episode + eps)

            # Reset environment if required (3)
            self.reset_env(exalearner)

            # Do the steps.  If batch_episode_frequency > 1 batch_steps_frequency == nsteps
            for i in range(self.batch_step_frequency):
                # Do inference (4)
                if ExaComm.env_comm.rank == 0:
                    action, policy_type = exalearner.agent.action(self.current_state)
                    if exalearner.action_type == "fixed":
                        action, policy_type = 0, -11

                # Set the step for envs that want to keep track
                exalearner.env.set_step_count(self.steps)

                # Broadcast action and do step (5 and 6)
                action = ExaComm.env_comm.bcast(action, root=0)
                next_state, reward, self.done, _ = exalearner.env.step(action)
                self.steps += 1

                # Clip rewards if specified in configuration
                if self.clip_rewards is not None:
                    reward = max(min(reward, self.clip_rewards[1]), self.clip_rewards[0])

                # Record experience (7)
                if ExaComm.env_comm.rank == 0:
                    exalearner.agent.remember(self.current_state, action, reward, next_state, self.done)
                    self.total_reward += reward

                # Check number of steps and broadcast (8)
                if self.steps == exalearner.nsteps:
                    self.done = True
                self.done = ExaComm.env_comm.bcast(self.done, 0)
                self.write_log(self.current_state, action, reward, next_state, self.total_reward, self.done, episode, self.steps, policy_type)

                # Update state (9)
                self.current_state = next_state

                if self.done:
                    break

        # Send batches back to the learner (10)
        if ExaComm.env_comm.rank == 0:
            batch_data = next(exalearner.agent.generate_data())
            self.send_batch(batch_data, policy_type, self.done, self.total_reward)
        return True

    def episode_round(self, exalearner):
        """
        Rounds to an even number of episodes for blocking purposes.
        We broadcast this result to everyone.  This is also a good
        sync point prior to running loops.

        Parameters
        ----------
        exalearner : ExaLearner
            This contains the agent and env
        """
        nepisodes = exalearner.nepisodes
        if ExaComm.global_comm.rank == 0:
            if self.block_size == ExaComm.agent_comm.size and ExaComm.agent_comm.size > 1:
                nactors = ExaComm.agent_comm.size - ExaComm.num_learners
                nactorBatch = nactors * self.batch_episode_frequency
                if nepisodes % nactorBatch:
                    nepisodes = (int(nepisodes / nactorBatch) + 1) * nactorBatch
            else:
                # This else should make sure nepisodes is factor of self.batch_episode_frequency
                # previous if makes sure of that.
                if nepisodes % self.batch_episode_frequency:
                    nepisodes = (int(nepisodes / self.batch_episode_frequency) + 1) * self.batch_episode_frequency

            # Just make it so everyone does at least one batch
            if nepisodes < (ExaComm.agent_comm.size - ExaComm.num_learners) * self.batch_episode_frequency:
                nepisodes = (ExaComm.agent_comm.size - ExaComm.num_learners) * self.batch_episode_frequency

            # For multi-learner, we must have equal amount of batches
            if ExaComm.num_learners > 1:
                # We can't guarantee cutoff wont leave unequal number of batches
                assert self.cutoff <= 0 or self.rolling_reward_length <= 1, "Cutoff not supported for multi-learner"
                nepisodes = (int(nepisodes / ExaComm.num_learners) + 1) * ExaComm.num_learners
        # This ensures everyone has the same nepisodes as well
        # as ensuring everyone is starting at the same time
        nepisodes = ExaComm.global_comm.bcast(nepisodes, 0)
        return nepisodes

    @PROFILE
    def run(self, exalearner):
        """
        This function is responsible for calling the appropriate initialization
        and looping over the actor/learner functions.  This function should
        be overloaded for more interesting workflows.

        We change the number of run episodes to make blocking easier.  When the
        block_size is set, we round up the number of episodes such that each
        actor will run the same number of episodes.

        Parameters
        ----------
        exalearner : ExaLearner
            This contains the agent and env
        """
        convergence = -1
        nepisodes = self.episode_round(exalearner)
        self.init_learner(exalearner)
        if ExaComm.is_agent():
            while self.alive and self.done_episode < nepisodes:
                self.actor(exalearner, nepisodes)
                do_convergence_check = self.learner(exalearner, nepisodes, 0)
                if do_convergence_check:
                    convergence = self.check_convergence()
                self.debug("Learner:", self.done_episode, nepisodes, do_convergence_check, convergence)
            # Send the done signal to the rest
            ExaComm.env_comm.bcast(self.done_episode, 0)
        else:
            keep_running = True
            while keep_running:
                keep_running = self.actor(exalearner, nepisodes)
                self.debug("Actor:", keep_running)

    def get_total_episodes_run(self):
        """
        The number of episodes finished.  This is important especially when using convergence cutoff.

        Returns
        -------
        int :
            Total number of episodes completed
        """
        return self.done_episode

    def get_total_reward(self):
        """
        Gives the sum of the rewards across episodes

        Returns
        -------
        int :
            Total reward across learning from all episodes
        """
        return sum(self.episode_reward_list)

    def get_rolling_reward(self):
        """
        Gives the rolling reward based on the configuration variable rolling_reward_length

        Returns
        -------
        int :
            Average reward of the rolling_reward_length number of episodes.
        """
        return np.mean(np.array(self.episode_reward_list[-self.rolling_reward_length:])), self.rolling_reward_length
