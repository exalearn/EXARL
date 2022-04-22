import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
import pickle
import exarl
from exarl.utils.globals import ExaGlobals
from exarl.utils.OUActionNoise import OUActionNoise
from exarl.utils.OUActionNoise import OUActionNoise2
logger = ExaGlobals.setup_logger(__name__)

@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


class DDPG_Vtrace(exarl.ExaAgent):
    is_learner: bool

    def __init__(self, env, is_learner):
        # Distributed variables
        self.is_learner = is_learner

        # Not used by agent but required by the learner class
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999

        # Environment space and action parameters
        self.env = env
        self.num_states = env.observation_space.shape[0]
        self.num_disc_actions = env.action_space.n
        # TODO: fix this later!! env.action_space.shape[0]
        self.num_actions = 1
        # self.upper_bound = env.action_space.high
        # self.lower_bound = env.action_space.low

        logger().info("Size of State Space:  {}".format(self.num_states))
        logger().info("Size of Action Space:  {}".format(self.num_actions))
        # logger().info('Env upper bounds: {}'.format(self.upper_bound))
        # logger().info('Env lower bounds: {}'.format(self.lower_bound))

        self.gamma = 0.99
        self.tau = 0.005

        std_dev = 0.2
        # ave_bound = (self.upper_bound + self.lower_bound) / 2
        # print('ave_bound: {}'.format(ave_bound))
        # self.ou_noise = OUActionNoise(mean=ave_bound, std_deviation=float(std_dev) * np.ones(1))

        # Experience data
        self.buffer_capacity = 5000
        self.batch_size = 64
        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_capacity, self.num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, self.num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros(
            (self.buffer_capacity, self.num_states))
        self.done_buffer = np.zeros((self.buffer_capacity, 1))
        self.memory = self.state_buffer  # BAD from the original code whoever wrote DDPG

        # Setup TF configuration to allow memory growth
        # tf.keras.backend.set_floatx('float64')
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(sess)

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
        critic_lr = 0.002
        actor_lr = 0.001
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

        # Vtrace
        self.time_step = -1
        self.n_steps = cd.run_params["n_steps"]
        # print("steps: ", self.n_steps)
        self.truncImpSampC = np.zeros(self.n_steps)
        self.truncImpSampR = np.zeros(self.n_steps)
        self.truncLevelC = 1
        self.truncLevelR = 1

        # should this be called somewhere?
        self.set_learner()

    def remember(self, state, action, reward, next_state, done):
        # If the counter exceeds the capacity then
        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = state
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.next_state_buffer[index] = next_state
        self.done_buffer[index] = int(done)
        self.buffer_counter += 1

    # @tf.function
    def update_grad(
            self,
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch):

        # print("curr_state_batch: ", state_batch)
        # print("next_state_batch: ", next_state_batch)

        # Training and updating Actor & Critic networks.
        with tf.GradientTape() as tape:

            curr_state_val = self.critic_model([state_batch], training=True)
            next_state_val = self.critic_model([next_state_batch])

            # print("curr_state_val: ", curr_state_val )
            # print("next_state_val: ", next_state_val )

            # TODO: compute the product of C, but here 1-step
            self.prodC = self.truncImpSampC[self.time_step]

            # print("prodC",          self.prodC)
            # print("truncImpSampR",  self.truncImpSampR[self.time_step])

            # Vtace target
            # TODO: need to sum for n-step backup
            # + \sum self.gamma * self.prodC * self.truncImpSampR[self.time_step] ...

            y = curr_state_val \
                + self.prodC * self.truncImpSampR[self.time_step] \
                * (reward_batch + self.gamma * next_state_val - curr_state_val)

            # print("y: ", y)

            critic_loss = tf.math.reduce_mean(
                tf.math.square(y - curr_state_val))

        logger().warning("Critic loss: {}".format(critic_loss))

        critic_grad = tape.gradient(
            critic_loss, self.critic_model.trainable_variables)

        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )

        # This code did not work
        tf.cast(action_batch, dtype=tf.uint8)
        # print("action_batch: ", action_batch)

        # """
        action_idx = [0 for i in range(self.batch_size)]

        for i in range(self.batch_size):
            action_idx[i] = int(action_batch[i].numpy())

        # print("action_idx: ", action_idx)

        with tf.GradientTape() as tape:

            output_behavi_actions = self.actor_model(
                state_batch, training=True)

            # print("output: ", output_behavi_actions)
            # TODO: update the next_state_val using the v-trace target
            # TODO: avoid using Python for loop
            actor_loss = 0
            for i in range(self.batch_size):
                actor_loss += self.truncLevelR * output_behavi_actions[action_idx[i]][0] \
                    * (reward_batch[action_idx[i]][0] + self.gamma
                       * next_state_val[action_idx[i]][0] - curr_state_val[action_idx[i]][0])

            actor_loss = actor_loss / self.batch_size

            """
            actor_loss = tf.math.reduce_mean(
                             self.truncLevelR * output_behavi_actions[action_idx][0] \
                             * ( reward_batch + self.gamma * next_state_val - curr_state_val ) )
                         # * log(output_behavi_actions[action_batch]) \
                         # * ( reward_batch + self.gamma * y - curr_state_val )
            """

            # critic_value = self.critic_model([state_batch], training=True)
            # actor_loss = -tf.math.reduce_mean(critic_value)

        # logger().warning("Actor loss: {}".format(actor_loss))
        actor_grad = tape.gradient(
            actor_loss, self.actor_model.trainable_variables)

        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )

    def get_actor(self):
        # State as input
        inputs = layers.Input(shape=(self.num_states, ))
        out = layers.Dense(256, activation="relu")(inputs)
        out = layers.Dense(256, activation="relu")(out)

        # out = layers.Dense(self.num_actions, activation="tanh", kernel_initializer=tf.random_uniform_initializer())(out)
        out = layers.Dense(self.num_disc_actions, activation="softmax")(out)

        # out = layers.Dense(self.num_actions, activation="sigmoid", kernel_initializer=tf.random_uniform_initializer())(out)

        # outputs = layers.Lambda(lambda i: i * self.upper_bound)(out)

        # model = tf.keras.Model(inputs, outputs)
        model = tf.keras.Model(inputs, out)

        return model

    def get_critic(self):

        # State as input
        state_input = layers.Input(shape=self.num_states)
        state_out = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.Dense(32, activation="relu")(state_out)

        # Action as input
        # action_input = layers.Input(shape=self.num_actions)
        # action_out = layers.Dense(32, activation="relu")(action_input)

        # Both are passed through seperate layer before concatenating
        # concat = layers.Concatenate()([state_out, action_out])

        # out = layers.Dense(256, activation="relu")(concat)

        out = layers.Dense(256, activation="relu")(state_out)
        out = layers.Dense(
            256,
            activation="linear",
            kernel_initializer=tf.random_uniform_initializer())(out)
        outputs = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input], outputs)

        return model

    def generate_data(self):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        logger().info('record_range:{}'.format(record_range))
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)
        logger().info('batch_indices:{}'.format(batch_indices))
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(
            self.next_state_buffer[batch_indices])

        yield state_batch, action_batch, reward_batch, next_state_batch

    def train(self, batch):
        # self.epsilon_adj()
        # if len(batch[0]) >= self.batch_size:
        #     logger().info('Training...')
        if self.is_learner:
            logger().warning('Training...')
            self.update_grad(batch[0], batch[1], batch[2], batch[3])

    def target_train(self):

        # Update the target model
        # if self.buffer_counter >= self.batch_size:
        # update_target(self.target_actor.variables, self.actor_model.variables, self.tau)
        # update_target(self.target_critic.variables, self.critic_model.variables, self.tau)

        model_weights = self.actor_model.get_weights()
        target_weights = self.target_actor.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = self.tau * model_weights[i] + \
                (1 - self.tau) * target_weights[i]

        self.target_actor.set_weights(target_weights)

        model_weights = self.critic_model.get_weights()
        target_weights = self.target_critic.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = self.tau * model_weights[i] + \
                (1 - self.tau) * target_weights[i]

        self.target_critic.set_weights(target_weights)

    def action(self, state):

        policy_type = 1
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)

        # Ai: changed behavior policy to choose the next action

        output_target_actions = tf.squeeze(self.target_actor(tf_state))
        # sampled_actions = tf.math.argmax(target_actions).numpy()

        output_behavi_actions = tf.squeeze(self.actor_model(tf_state))
        sampled_actions = tf.math.argmax(output_behavi_actions).numpy()

        # TODO: make it available for mutlti-dimensions

        # noise = self.ou_noise()
        # print("sampled_actions: ", sampled_actions)

        sampled_actions_wn = sampled_actions  # .numpy() # + noise
        legal_action = sampled_actions_wn     # Ai: what is this for?

        # print("legal_action: ", legal_action)

        isValid = self.env.action_space.contains(sampled_actions_wn)
        if not isValid:
            # legal_action = np.random.uniform(low=self.lower_bound, high=self.upper_bound, size=(self.num_actions,))
            legal_action = random.randint(0, self.num_disc_actions - 1)
            policy_type = 0
            logger().warning(
                'Bad action: {}; Replaced with: {}'.format(
                    sampled_actions_wn, legal_action))
            # logger().warning('Policy action: {}; noise: {}'.format(sampled_actions,noise))

        return_action = [np.squeeze(legal_action)]
        logger().warning('Legal action:{}'.format(return_action))

        # ************************** computations for vtrace ******************
        # TODO: make a function for this procedure

        # compute prob for target and behvior policy for (action|state)
        sum_target_val = 0.0
        sum_behavi_val = 0.0

        min_target_val = min(output_target_actions.numpy())
        min_behavi_val = min(output_behavi_actions.numpy())

        # print("min_target_val", min_target_val)

        for i in range(self.num_disc_actions):
            # if ( min_target_val < 0):
            # + abs(min_target_val)
            sum_target_val += output_target_actions[i].numpy()
            # if ( min_behavi_val < 0):
            # + abs(min_behavi_val)
            sum_behavi_val += output_behavi_actions[i].numpy()

        # prob_target_action = output_target_actions[return_action].numpy() / sum_target_val
        # prob_behavi_action = output_behavi_actions[return_action].numpy() / sum_behavi_val

        prob_target_action = output_target_actions[return_action].numpy()
        prob_behavi_action = output_behavi_actions[return_action].numpy()

        if prob_target_action < 0:
            prob_target_action = 0.000001
        if prob_behavi_action < 0:
            prob_behavi_action = 0.000001

        # printint("prob_target_action: ", prob_target_action)
        # print("prob_behavi_action: ", prob_behavi_action)
        # print("IS ratio: ",           prob_target_action / prob_behavi_action)

        self.truncImpSampC[self.time_step] = min(
            self.truncLevelC, prob_target_action / prob_behavi_action)
        self.truncImpSampR[self.time_step] = min(
            self.truncLevelR, prob_target_action / prob_behavi_action)

        if (self.time_step == self.n_steps - 1):
            self.time_step = -1

        # return return_action[0], policy_type
        return legal_action, policy_type

    # For distributed actors #
    def get_weights(self):
        return self.target_actor.get_weights()

    def set_weights(self, weights):
        self.target_actor.set_weights(weights)

    def set_learner(self):
        # print("##############")
        # print("set_learner")
        # print("##############")
        self.is_learner = True
        self.actor_model = self.get_actor()
        self.critic_model = self.get_critic()
        self.target_actor = self.get_actor()
        self.target_critic = self.get_critic()
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

    # Extra methods
    def update(self):
        print("Implement update method in ddpg.py")

    def load(self, filename):
        layers = self.target_actor.layers
        with open(filename, "rb") as f:
            pickle_list = pickle.load(f)

        for layerId in range(len(layers)):
            assert layers[layerId].name == pickle_list[layerId][0]
            layers[layerId].set_weights(pickle_list[layerId][1])

    def save(self, filename):
        layers = self.target_actor.layers
        pickle_list = []
        for layerId in range(len(layers)):
            weigths = layers[layerId].get_weights()
            pickle_list.append([layers[layerId].name, weigths])

        with open(filename, "wb") as f:
            pickle.dump(pickle_list, f, -1)

    def monitor(self):
        print("Implement monitor method in ddpg.py")

    def set_agent(self):
        print("Implement set_agent method in ddpg.py")

    # def print_timers(self):
    #     print("Implement print_timers method in ddpg.py")

    def epsilon_adj(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
