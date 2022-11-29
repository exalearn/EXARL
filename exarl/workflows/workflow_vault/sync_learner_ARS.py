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

class SYNC_ARS(exarl.ExaWorkflow):
    """
    This class implements a workflow by breaking the functionality into pieces.
    We define 3 key terms used throughout the implementation/documentation:

    Learner - the rank responsible for running the train/target train functions.
    Agents - the ranks responsible for training/inference.  Agents include learners.
    Actors - everyone that is not a learner.

    The following is an example useful for understanding the above descriptions.
    The example assumes 14 ranks (2 Learners, 5 Agents, 4 Environment)
    We present both the comms and the ranks.
    There is no actor comm so we add * to depict actors.
    Rank  0  1  2  3  4  5  6  7  8  9  10 11 12 13
    Learn     0  1  -  -  -  -  -  -  -  -  -  -  -  -
    Agent     0  1  2  -  -  -  3  -  -  -  4  -  -  -
    Actor     -  -  *  *  *  *  *  *  *  *  *  *  *  *
    Envir     -  -  0  1  2  3  0  1  2  3  0  1  2  3

    For a single rank execution, global rank 0 is the learner, agent, and environment.

    The cut-off condition for this is based on how many episodes the
    done learner observes.  We have two sets of internal variables
    one set used by the learners and another used by the actors.
    Batches and weights are passed via the leaner and actor calls by
    setting self.batch and self.weights.

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

    batch_frequency : int
        This value is used to determine how often we should
        send a batch of data.  The value represents performing
        batch_frequency steps per 1 batch send.

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

    verbose : bool
        Debug print flag
    """
    verbose = False

    def __init__(self):
        self.debug('Creating SYNC', ExaComm.global_comm.rank, ExaComm.is_learner(), ExaComm.is_agent(), ExaComm.is_actor())
        # print("Block:", ExaGlobals.lookup_params('episode_block'), "Batch Step Frequency:", ExaGlobals.lookup_params('batch_step_frequency'), "Train Frequency:", ExaGlobals.lookup_params('train_frequency'), "Batch Size:", ExaGlobals.lookup_params('batch_size'), flush=True)
        
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

        # How often to update target parameters
        self.update_target_frequency = ExaGlobals.lookup_params('update_target_frequency')

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
        if ExaComm.is_agent():
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
        if SYNC_ARS.verbose:
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
    @PROFILE
    def write_log(self, current_state, action, reward, next_state, total_reward, done, episode, steps, policy_type, epsilon):
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

        steps : int
            This is the current step within an episode

        policy_type : int
            This value is given by the action...
            TODO: Make this comment better

        epsilon : float
            The current value for given agent...
            TODO: Make this comment better
        """
        if ExaComm.is_agent():
            start_ = time.time()
            self.data_matrix.append([start_, current_state, action, reward, next_state, total_reward, done, episode, steps, policy_type, epsilon])
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

    @introspect
    def send_model(self, exalearner, episode, train_return, dst):
        """
        This function is responsible for sending the model from the learner to
        other agents.  For the sync learner, we just store the weights in the workflow.
        For more interesting workflows, this should include an MPI send or RMA operation.
        We intend for this function is to be overloaded in subsequent workflows.

        The workflow expects a message containing the episode, epsilon, and the model weights.
        To use the learner and actor functions, this must be respected.  Otherwise, it
        one will need to rewrite those functions in a derived class.

        Parameters
        ----------
        exalearner : ExaLearner
            This contains the agent and env

        episode : int
            The current episode curresponding to the model generation

        train_return : list
            This is what comes out of the learner calling train to be sent back
            to the actor (i.e. indices and losses).

        dst : int
            This is the destination rank given by the agent communicator
        """
        weights = exalearner.agent.get_weights()
        self.weights = [episode, exalearner.agent.epsilon, weights]
        if train_return:
            self.weights.append(train_return)

    @introspect
    def recv_model(self):
        """
        This function is the corresponding receive function to
        the send_model function.  Here the weights are being received by the
        the other agents (sent from the learner).  Again for the sync learner
        we retrieve this data which is stored locally, however this function is
        to be overloaded for more interesting workflows.

        Returns
        ----------
        list :
            This list should contain the episode, epsilon, model weights,
            and the train return (indices and losses if turned on)
        """
        return self.weights

    @introspect
    def send_batch(self, batch_data, policy_type, done, epsilon, episode_reward):
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
            TODO: Make this description better
        """
        self.batch = [ExaComm.agent_comm.rank, batch_data, policy_type, done, epsilon, episode_reward]

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

    def get_average_reward(self, rolling_reward_length=None):
        if rolling_reward_length is None:
            return np.mean(np.array(self.episode_reward_list))
        return np.mean(np.array(self.episode_reward_list[-rolling_reward_length:]))

    def check_convergence(self, nepisodes):
        # Lets us know how we are doing
        if self.cutoff > 0 and self.rolling_reward_length > 1 and not self.converged:
            if len(self.episode_reward_list) >= self.rolling_reward_length:
                ave = np.mean(np.abs(np.diff(np.array(self.episode_reward_list[-self.rolling_reward_length:]))))
                # print("Check:", ave, len(self.episode_reward_list), self.episode_reward_list[-5:]) #self.episode_reward_list[-self.rolling_reward_length:])
                if ave < self.cutoff:
                    self.converged = True
                    print("Converged:", len(self.episode_reward_list), "Alive:", self.alive, "Ave:", ave, "Last:", self.episode_reward_list[-1])
                return ave
        return -1

    # @introspect
    # @PROFILE
    def learner(self, exalearner, nepisodes, start_rank):
        """
        This function is performed by the learner.  The learner
        performs the following key steps:

        1. Receives batches of experiences
        2. Trains the models on the data received
        3. Checks if an episode has finished
        4. Sends data back to the appropriate actors

        Each call to train/target train represents a new model
        generation.  On-policy learning is means that the experiences
        contained by the batch are from the previous model (i.e.
        there is only one generation of models between them).
        Off-policy learning is done when experiences are used from
        previous model generations to train.  Training with a
        single actor will result in on-policy learning.  When we
        scale to use multiple actors we will by definition be
        training with older models.  If we were to collect data
        from each actor round robin with N actors, the model would
        be off policy by N models.

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
            src, batch, policy_type, done, epsilon, total_reward = self.recv_batch()
            self.train_return[src] = exalearner.agent.train(batch)

            if self.model_count % self.update_target_frequency == 0:
                exalearner.agent.update_target()

            self.model_count += 1
            to_send.append(src)

            exalearner.agent.epsilon = min(exalearner.agent.epsilon, epsilon)

            if done:
                self.episode_reward_list.append(total_reward)
                self.done_episode += self.batch_episode_frequency
                self.episode_per_rank[src] = self.next_episode
                self.next_episode += self.batch_episode_frequency
                ret = True
            
            if self.converged:
                self.episode_per_rank[src] = nepisodes
        for dst in to_send:
            self.send_model(exalearner, self.episode_per_rank[dst], self.train_return[dst], dst)
            if self.episode_per_rank[dst] >= nepisodes:
                self.alive -= 1

        self.save_weights(exalearner, self.done_episode, nepisodes)
        return ret

    def actor_recv(self, exalearner, nepisodes):
        # Get model and update the other env ranks (1)
        if ExaComm.env_comm.rank == 0:
            episode, epsilon, weights, *train_ret = self.recv_model()
        episode = ExaComm.env_comm.bcast(episode, 0)
        
        return episode,epsilon,weights,train_ret


    # @introspect
    # @PROFILE
    def actor(self, exalearner, nepisodes,comm_flag=True):
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
        epsilon = 0
        action = 0
        policy_type = 0

        
        if comm_flag:
            episode,epsilon,weights,train_ret = self.actor_recv(exalearner,nepisodes)
        else:
            weights = exalearner.agent.get_weights()
            epsilon = exalearner.agent.epsilon
            train_ret=[]

        # Check if you are already crossed the required number of episodes...
        if episode >= nepisodes:
            return False

        # Set agent for rank 0 (2)
        if ExaComm.env_comm.rank == 0:
            exalearner.agent.epsilon = epsilon
            exalearner.agent.set_weights(weights)
            if train_ret:
                # JS: This call flattens the list from *train_ret above
                train_ret = [item for sublist in train_ret for item in sublist]
                exalearner.agent.set_priorities(*train_ret)
        
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

                # Clip rewards if specified
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
                reward = reward[0]
                action = action[0]
                self.total_reward = self.total_reward[0]
                # print(type(self.current_state), type(next_state), ".....")

                # print(self.current_state, action, reward, next_state, self.total_reward, self.done, episode, self.steps, policy_type, epsilon)
                self.write_log(self.current_state, action, reward, next_state, self.total_reward, self.done, episode, self.steps, policy_type, epsilon)

                # Update state (9)
                self.current_state = next_state

                if self.done:
                    break
        if comm_flag:
            self.actor_send(exalearner,policy_type=policy_type)

        return True

    def actor_send(self,exalearner,policy_type=None):
         # Send batches back to the learner (10)
        if ExaComm.env_comm.rank == 0:
            batch_data = next(exalearner.agent.generate_data())
            episode_reward_rollout = batch_data['rollout/ep_rew_mean']
            self.send_batch(batch_data, policy_type, self.done, exalearner.agent.epsilon, episode_reward_rollout)
        return

    def episode_round(self, exalearner):
        """
        Rounds to an even number of episodes for blocking purposes.
        We broadcast this result to everyone.  This is also a good
        sync point prior to running loops.

        This function also adds the members workflow_episode and
        workflow_step to the environment for environments who
        want to keep track of the current episode/step.

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
        print("Num eps:", nepisodes, self.batch_episode_frequency, self.batch_step_frequency)
        # This ensures everyone has the same nepisodes as well
        # as ensuring everyone is starting at the same time
        nepisodes = ExaComm.global_comm.bcast(nepisodes, 0)
        return nepisodes

    def ARS_actorWrappper(self,exalearner,nepisodes):

        # Populate the +ve and -ve weights.. 
        pm_W = exalearner.agent.Generate_pmW()
        
        # Loop over all the weights and 
        for i,Ws in enumerate(pm_W):
            # print(f" Weights-{i} send from workflow:, {Ws}")
            # set the weights.... in the agent
            exalearner.agent.set_weights(Ws)
            
                
            # Run the actor until we finish the n-steps of the environment.
            self.actor(exalearner, nepisodes,comm_flag=False)
            # print("Actor Call Finish..")
        
        # print("Calling Wrapper Actor Send....")
        # Send the data collected for all +/- weights 
        # environment rollout to the learner for policy update
        self.actor_send(exalearner)
        
        return True

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
                self.ARS_actorWrappper(exalearner,nepisodes)
                do_convergence_check = self.learner(exalearner, nepisodes, 0)
                if do_convergence_check:
                    convergence = self.check_convergence(nepisodes)
                self.debug("Learner:", self.done_episode, nepisodes, do_convergence_check, convergence)
            # Send the done signal to the rest
            ExaComm.env_comm.bcast(self.done_episode, 0)
            rolling_reward_length = self.rolling_reward_length
            
            if rolling_reward_length < 1 and rolling_reward_length <= self.done_episode:
                rolling_reward_length = None
            try:
                print(self.env.all_seeds)
            except:
                pass
            print("Done episodes:", self.done_episode, "Rolling Reward:", rolling_reward_length, "Average Total Reward:", self.get_average_reward(rolling_reward_length))
        else:
            keep_running = True
            while keep_running:
                keep_running = self.actor(exalearner, nepisodes)
                self.debug("Actor:", keep_running)
