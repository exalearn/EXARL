import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
import os
from datetime import datetime
from utils.OUActionNoise import OUActionNoise
from utils.OUActionNoise import OUActionNoise2
from utils.introspect import introspectTrace

import exarl as erl

import utils.log as log
from utils.candleDriver import initialize_parameters
run_params = initialize_parameters()
logger = log.setup_logger(__name__, run_params['log_level'])


@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


class DDPG(erl.ExaAgent):
    is_learner: bool

    def __init__(self, env):
        # Distributed variables
        self.is_learner = False

        # Not used by agent but required by the learner class
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999

        # Environment space and action parameters
        self.env = env
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.upper_bound = env.action_space.high
        self.lower_bound = env.action_space.low

        logger.info("Size of State Space:  {}".format(self.num_states))
        logger.info("Size of Action Space:  {}".format(self.num_actions))
        logger.info('Env upper bounds: {}'.format(self.upper_bound))
        logger.info('Env lower bounds: {}'.format(self.lower_bound))

        self.gamma = 0.995
        self.tau = 0.05

        # start_std = self.upper_bound - self.lower_bound
        # stop_std = 0.05*start_std
        # self.ou_noise = OUActionNoise2(mean=float(0) * np.ones(1),
        #                               start_std=float(start_std) * np.ones(1),
        #                               stop_std=float(stop_std) * np.ones(1),
        #                               damping=0.0005)

        std_dev = 0.2
        self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

        # Experience data
        self.buffer_capacity = 5000
        self.batch_size = 64
        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_capacity, self.num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, self.num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, self.num_states))
        self.done_buffer = np.zeros((self.buffer_capacity, 1))
        self.memory = self.state_buffer  # BAD

        # TODO: Required by the learner
        self.actor_model = self.get_actor()
        self.critic_model = self.get_critic()

        # Every agent needs this
        self.target_critic = self.get_critic()
        self.target_actor = self.get_actor()

        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        # Learning rate for actor-critic models
        critic_lr = 0.0001
        actor_lr = 0.002
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

    def remember(self, state, action, reward, next_state, done):
        # If the counter exceeds the capacity then
        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = state
        self.action_buffer[index] = action[0]
        self.reward_buffer[index] = reward
        self.next_state_buffer[index] = next_state
        self.done_buffer[index] = int(done)
        self.buffer_counter += 1

    # @tf.function
    def update_grad(self, state_batch, action_batch, reward_batch, next_state_batch):
        # Training and updating Actor & Critic networks.
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + self.gamma * self.target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        logger.info("loss: {}".format(critic_loss))
        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            critic_value = self.critic_model([state_batch, actions], training=True)
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )

    def get_actor(self):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(self.num_states,))
        out = layers.Dense(256, activation="relu")(inputs)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

        outputs = outputs * self.upper_bound
        model = tf.keras.Model(inputs, outputs)
        return model

    def get_critic(self):
        # State as input
        state_input = layers.Input(shape=self.num_states)
        state_out = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.Dense(32, activation="relu")(state_out)

        # Action as input
        action_input = layers.Input(shape=self.num_actions)
        action_out = layers.Dense(32, activation="relu")(action_input)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(256, activation="relu")(concat)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model

    @introspectTrace()
    def generate_data(self):
        # if self.buffer_counter < self.batch_size:
        #      yield [], [], [], []
        record_range = min(self.buffer_counter, self.buffer_capacity)
        logger.info('record_range:{}'.format(record_range))
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)
        logger.info('batch_indices:{}'.format(batch_indices))
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        yield state_batch, action_batch, reward_batch, next_state_batch

    @introspectTrace()
    def train(self, batch):
        # self.epsilon_adj()
        # if len(batch[0]) >= self.batch_size:
        #     logger.info('Training...')
        self.update_grad(batch[0], batch[1], batch[2], batch[3])

    @introspectTrace()
    def target_train(self):
        # Update the target model
        # if self.buffer_counter >= self.batch_size:
        # update_target(self.target_actor.variables, self.actor_model.variables, self.tau)
        # update_target(self.target_critic.variables, self.critic_model.variables, self.tau)
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

    @introspectTrace()
    def action(self, state):
        # random.seed(datetime.now())
        # random_data = os.urandom(4)
        # np.random.seed(int.from_bytes(random_data, byteorder="big"))
        # rdm = np.random.rand()
        # if rdm <= self.epsilon:
        #     action = np.random.uniform(low=self.lower_bound, high=self.upper_bound, size=self.env.action_space.shape)
        #     # random.randrange(self.env.action_space.n)
        #     logger.info('rdm action:{}'.format(action))
        #     return action, 0
        # else:
        # random.seed(datetime.now())
        # random_data = os.urandom(4)
        # np.random.seed(int.from_bytes(random_data, byteorder="big"))
        # if self.buffer_counter <= self.batch_size:
        #     action = np.random.uniform(low=self.lower_bound, high=self.upper_bound, size=(1,))
        #     return action, 0
        # else:
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        sampled_actions = tf.squeeze(self.target_actor(tf_state))
        # sampled_actions = tf.squeeze(self.actor_model(tf_state))
        noise = self.ou_noise()
        # Adding noise to action
        sampled_actions_wn = sampled_actions.numpy() + noise
        # Make sure action is within bounds
        legal_action = np.clip(sampled_actions_wn, self.lower_bound, self.upper_bound)
        logger.info('legal action:{}'.format([np.squeeze(legal_action)]))
        # return legal_action, noise[0]
        return [np.squeeze(legal_action)], 1

    # For distributed actors #
    # @introspectTrace()
    def get_weights(self):
        return self.target_actor.get_weights()

    # @introspectTrace()
    def set_weights(self, weights):
        self.target_actor.set_weights(weights)

    def set_learner(self):
        self.is_learner = True

    # Extra methods
    def update(self):
        print("Implement update method in ddpg.py")

    def load(self):
        print("Implement load method in ddpg.py")

    def save(self, results_dir):
        print("Implement load method in ddpg.py")

    def monitor(self):
        print("Implement monitor method in ddpg.py")

    def set_agent(self):
        print("Implement set_agent method in ddpg.py")

    def print_timers(self):
        print("Implement print_timers method in ddpg.py")

    def epsilon_adj(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
