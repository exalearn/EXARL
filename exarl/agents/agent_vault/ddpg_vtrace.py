import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import exarl
from exarl.utils.globals import ExaGlobals
logger = ExaGlobals.setup_logger(__name__)

@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


class DDPG_Vtrace(exarl.ExaAgent):
    """Deep deterministic policy gradient agent.
    Inherits from ExaAgent base class.
    """
    is_learner: bool

    def __init__(self, env, is_learner):
        """DDPG_Vtrace constructor

        Args:
            env (OpenAI Gym environment object): env object indicates the RL environment
            is_learner (bool): Used to indicate if the agent is a learner or an actor
        """
        # Distributed variables
        self.is_learner = is_learner

        # Environment space and action parameters
        self.env = env
        self.num_states = env.observation_space.shape[0]
        self.num_disc_actions = env.action_space.n

        # TODO: fix this later!! env.action_space.shape[0]
        self.num_actions = 1

        logger().info("Size of State Space:  {}".format(self.num_states))
        logger().info("Size of Action Space:  {}".format(self.num_actions))

        # model definitions
        self.actor_dense = ExaGlobals.lookup_params('actor_dense')
        self.actor_dense_act = ExaGlobals.lookup_params('actor_dense_act')
        self.actor_out_act = ExaGlobals.lookup_params('actor_out_act')
        self.actor_optimizer = ExaGlobals.lookup_params('actor_optimizer')
        self.critic_state_dense = ExaGlobals.lookup_params('critic_state_dense')
        self.critic_state_dense_act = ExaGlobals.lookup_params('critic_state_dense_act')
        self.critic_action_dense = ExaGlobals.lookup_params('critic_action_dense')
        self.critic_action_dense_act = ExaGlobals.lookup_params('critic_action_dense_act')
        self.critic_concat_dense = ExaGlobals.lookup_params('critic_concat_dense')
        self.critic_concat_dense_act = ExaGlobals.lookup_params('critic_concat_dense_act')
        self.critic_out_act = ExaGlobals.lookup_params('critic_out_act')
        self.critic_optimizer = ExaGlobals.lookup_params('critic_optimizer')
        self.tau = ExaGlobals.lookup_params('tau')
        self.gamma = ExaGlobals.lookup_params('gamma')

        # Not used by agent but required by the learner class
        self.epsilon = ExaGlobals.lookup_params('epsilon')
        self.epsilon_min = ExaGlobals.lookup_params('epsilon_min')
        self.epsilon_decay = ExaGlobals.lookup_params('epsilon_decay')

        # Experience data
        self.buffer_capacity = ExaGlobals.lookup_params('buffer_capacity')
        self.batch_size = ExaGlobals.lookup_params('batch_size')
        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_capacity, self.num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, self.num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, self.num_states))
        self.done_buffer = np.zeros((self.buffer_capacity, 1))
        self.memory = self.state_buffer  # BAD from the original DDPG code

        # Setup TF configuration to allow memory growth
        # tf.keras.backend.set_floatx('float64')
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(sess)
        # TODO: The name tf.keras.backend.set_session is deprecated. Please use tf.compat.v1.keras.backend.set_session instead.

        # Training model only required by the learners
        self.actor_model = None
        self.critic_model = None
        if self.is_learner:
            self.actor_model = self.get_actor()
            self.critic_model = self.get_critic()

        # Every agent needs this, however, actors only use the CPU (for now)
        self.target_critic = None
        self.target_actor = None
        if self.is_learner:
            self.target_actor = self.get_actor()
            self.target_critic = self.get_critic()
            self.target_actor.set_weights(self.actor_model.get_weights())
            self.target_critic.set_weights(self.critic_model.get_weights())
        else:
            with tf.device('/CPU:0'):
                self.target_actor = self.get_actor()
                self.target_critic = self.get_critic()

        # Learning rate for actor-critic models
        self.critic_lr = ExaGlobals.lookup_params('critic_lr')
        self.actor_lr = ExaGlobals.lookup_params('actor_lr')

        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)

        # Vtrace
        self.time_step = -1   # TODO: can we get the current time-step from the base class?
        self.truncLevelC = ExaGlobals.lookup_params('truncLevelC')
        self.truncLevelR = ExaGlobals.lookup_params('truncLevelR')
        self.n_step_backup = ExaGlobals.lookup_params('n_step_backup')
        self.n_steps = ExaGlobals.lookup_params('n_steps')
        self.truncImpSampC = np.zeros(self.n_steps)
        self.truncImpSampR = np.zeros(self.n_steps)
        # TODO: need to reset truncImpSampC and truncLevelR in every episode

    def remember(self, state, action, reward, next_state, done):
        """Add experience to replay buffer

        Args:
            state (list or array): Current state of the system
            action (list or array): Action to take
            reward (list or array): Environment reward
            next_state (list or array): Next state of the system
            done (bool): Indicates episode completion
        """
        # If the counter exceeds the capacity then
        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = state
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.next_state_buffer[index] = next_state
        self.done_buffer[index] = int(done)
        self.buffer_counter += 1

    # @tf.function
    def update_grad(self, state_batch, action_batch, reward_batch, next_state_batch):
        """Update gradients - training step

        Args:
            state_batch (list): list of states
            action_batch (list): list of actions
            reward_batch (list): list of rewards
            next_state_batch (list): list of next states
        """

        logger().info("curr_state_batch: {}".format(state_batch))
        logger().info("next_state_batch: {}".format(next_state_batch))

        # Training and updating Actor & Critic networks.
        with tf.GradientTape() as tape:

            curr_state_val = self.critic_model([state_batch], training=True)
            next_state_val = self.critic_model([next_state_batch], training=True)

            logger().info("curr_state_val: {}".format(curr_state_val))
            logger().info("next_state_val: {}".format(next_state_val))

            # calculate Vtace target
            # TODO: compute the product of C, but here 1-step
            # TODO: need to sum for n-step backup

            self.prodC = self.truncImpSampC[self.time_step]
            temp_diff = self.truncImpSampR[self.time_step] * \
                           (reward_batch + self.gamma * next_state_val - curr_state_val)
            y = curr_state_val + self.prodC * temp_diff

            """ n-step back-up v-trace
            for t in range(self.time_step, self.time_step+self.n_step_backup+1):
                self.prodC = self.truncImpSampC[t]

            for t in range(self.time_step, self.time_step+self.n_step_backup+1):
                temp_diff[t] = self.truncImpSampR[self.time_step] *
                               (reward_batch[t] + self.gamma * state_val[t+1] - state_val[t])
                vtrace_rhs = pow(self.gamma, t-self.time_step) * self.prodC * temp_diff[t]

            y = curr_state_val + vtrace_rhs
            """

            logger().info("prodC: {}".format(self.prodC))
            logger().info("truncImpSampR: {}".format(self.truncImpSampR[self.time_step]))
            logger().info("y: {}".format(y))

            critic_value = self.critic_model([state_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        logger().warning("Critic loss: {}".format(critic_loss))
        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:

            output_behavi_actions = self.actor_model(state_batch, training=True)
            action_log_probs = tf.math.log(output_behavi_actions)
            actor_loss = - self.truncLevelR * tf.math.reduce_sum(action_log_probs)

        logger().warning("Actor loss: {}".format(actor_loss))
        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )

    def get_actor(self):
        """Define actor network

        Returns:
            model: actor model
        """
        # State as input
        inputs = layers.Input(shape=(self.num_states,))
        # first layer takes inputs
        out = layers.Dense(self.actor_dense[0], activation=self.actor_dense_act)(inputs)
        # loop over remaining layers
        for i in range(1, len(self.actor_dense)):
            out = layers.Dense(self.actor_dense[i], activation=self.actor_dense_act)(out)
        # output layer has dimension actions, separate activation setting
        out = layers.Dense(self.num_disc_actions, activation=self.actor_out_act,
                           kernel_initializer=tf.random_uniform_initializer())(out)
        model = tf.keras.Model(inputs, out)

        return model

    def get_critic(self):
        """Define critic network

        Returns:
            model: critic network
        """
        # State as input
        state_input = layers.Input(shape=self.num_states)
        # first layer takes inputs
        state_out = layers.Dense(self.critic_state_dense[0],
                                 activation=self.critic_state_dense_act)(state_input)
        # loop over remaining layers
        for i in range(1, len(self.critic_state_dense)):
            state_out = layers.Dense(self.critic_state_dense[i],
                                     activation=self.critic_state_dense_act)(state_out)

        out = layers.Dense(self.critic_concat_dense[0], activation=self.critic_state_dense_act)(state_out)
        out = layers.Dense(self.critic_concat_dense[0], activation=self.critic_out_act,
            kernel_initializer=tf.random_uniform_initializer())(out)
        outputs = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input], outputs)

        return model

    def has_data(self):
        """Indicates if the buffer has data

        Returns:
            bool: True if buffer has data
        """
        return (self.buffer_counter > 0)

    def generate_data(self):
        """Generate data for training

        Yields:
            state_batch (list): list of states
            action_batch (list): list of actions
            reward_batch (list): list of rewards
            next_state_batch (list): list of next states
        """
        if self.has_data():
            record_range = min(self.buffer_counter, self.buffer_capacity)
            logger().info('record_range:{}'.format(record_range))
            # Randomly sample indices
            batch_indices = np.random.choice(record_range, self.batch_size)
        else:
            batch_indices = [0] * self.batch_size

        logger().info('batch_indices:{}'.format(batch_indices))
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        yield state_batch, action_batch, reward_batch, next_state_batch

    def train(self, batch):
        """Train the NN

        Args:
            batch (list): sampled batch of experiences
        """
        if self.is_learner:
            logger().warning('Training...')
            self.update_grad(batch[0], batch[1], batch[2], batch[3])
        else:
            logger().warning('Why is is_learner false...')

    def update_target(self):
        """Update target model
        """
        # Update the target model
        model_weights = self.actor_model.get_weights()
        target_weights = self.target_actor.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = self.tau * model_weights[i] + (1 - self.tau) * target_weights[i]
        self.target_actor.set_weights(target_weights)

        model_weights = self.critic_model.get_weights()
        target_weights = self.target_critic.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = self.tau * model_weights[i] + (1 - self.tau) * target_weights[i]
        self.target_critic.set_weights(target_weights)

    def action(self, state):
        """Returns sampled action

        Args:
            state (list or array): Current state of the system

        Returns:
            action (list or array): Action to take
            policy (int): random (0) or inference (1)
        """
        policy_type = 1
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)

        # changed behavior policy to choose the next action

        self.output_target_actions = tf.squeeze(self.target_actor(tf_state))
        # sampled_actions = tf.math.argmax(target_actions).numpy()

        self.output_behavi_actions = self.actor_model(tf_state)
        # sampled_actions = tf.math.argmax(self.output_behavi_actions).numpy()

        # Sample next action from the action probability distribution
        sampled_action = tf.random.categorical(self.output_behavi_actions, 1)[0, 0]
        # action_probs_t = tf.nn.softmax(self.output_behavi_actions)

        self.return_action = sampled_action.numpy()
        
        self.output_behavi_actions = tf.squeeze(self.output_behavi_actions)
        
        # logger().info('sampled_actions: {}'.format(sampled_actions))
        logger().info('return_action: {}'.format(self.return_action))

        self.time_step += 1

        self.setTruncatedImportantSamplingRatios()

        if (self.time_step == self.n_steps - 1): self.time_step = -1

        return self.return_action, policy_type

    def setTruncatedImportantSamplingRatios(self):
        """ set trunctated important sampling ratios (truncImpSampC and truncImpSampR)
        """

        prob_target_action = self.output_target_actions[self.return_action].numpy()
        prob_behavi_action = self.output_behavi_actions[self.return_action].numpy()

        # if prob_target_action < 0: prob_target_action = 0.000001
        # if prob_behavi_action < 0: prob_behavi_action = 0.000001

        # importance sampling ratio
        is_ratio = prob_target_action / prob_behavi_action

        self.truncImpSampC[self.time_step] = min(self.truncLevelC, is_ratio)
        self.truncImpSampR[self.time_step] = min(self.truncLevelR, is_ratio)

        logger().info('prob_target_action: {}'.format(prob_target_action))
        logger().info('prob_behavi_action: {}'.format(prob_behavi_action))
        logger().info('IS ratio: {}'.format(is_ratio))
        logger().info('truncImpSampC: {}'.format(self.truncImpSampC))
        logger().info('truncImpSampR: {}'.format(self.truncImpSampR))

    # For distributed actors #
    def get_weights(self):
        """Get weights from target model

        Returns:
            weights (list): target model weights
        """
        return self.target_actor.get_weights()

    def set_weights(self, weights):
        """Set model weights

        Args:
            weights (list): model weights
        """
        self.target_actor.set_weights(weights)

    # Extra methods
    def update(self):
        print("Implement update method in ddpg.py")

    def epsilon_adj(self):
        """Update epsilon value
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def set_priorities(self, indices, loss):
        # TODO implement this
        pass
