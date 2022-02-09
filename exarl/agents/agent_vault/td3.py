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
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import random
import os
import pickle
from datetime import datetime
from exarl.utils.OUActionNoise import OUActionNoise
from exarl.utils.OUActionNoise import OUActionNoise2
from exarl.utils.memory_type import MEMORY_TYPE
# from ._networks_td3 import ActorModel, CriticModel
from ._replay_buffer import ReplayBuffer, HindsightExperienceReplayMemory, PrioritedReplayBuffer

import exarl as erl

import exarl.utils.log as log
import exarl.utils.candleDriver as cd
logger = log.setup_logger(__name__, cd.run_params['log_level'])


class TD3(erl.ExaAgent):

    def __init__(self, env, is_learner=False, update_actor_iter=2):
        # Distributed variables
        self.is_learner = is_learner

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

        self.gamma = cd.run_params['gamma']
        self.tau = cd.run_params['tau']

        # model definitions
        # model definitions
        self.actor_dense = cd.run_params['actor_dense']
        self.actor_dense_act = cd.run_params['actor_dense_act']
        self.actor_out_act = cd.run_params['actor_out_act']
        self.actor_optimizer = cd.run_params['actor_optimizer']
        self.critic_state_dense = cd.run_params['critic_state_dense']
        self.critic_state_dense_act = cd.run_params['critic_state_dense_act']
        self.critic_action_dense = cd.run_params['critic_action_dense']
        self.critic_action_dense_act = cd.run_params['critic_action_dense_act']
        self.critic_concat_dense = cd.run_params['critic_concat_dense']
        self.critic_concat_dense_act = cd.run_params['critic_concat_dense_act']
        self.critic_out_act = cd.run_params['critic_out_act']
        self.critic_optimizer = cd.run_params['critic_optimizer']
        self.replay_buffer_type = cd.run_params['replay_buffer_type']

        std_dev = 0.2
        # ave_bound = (self.upper_bound + self.lower_bound) / 2
        ave_bound = np.zeros(1)
        print('ave_bound: {}'.format(ave_bound))
        self.ou_noise = OUActionNoise(mean=ave_bound, std_deviation=float(std_dev) * np.ones(1))

        # Not used by agent but required by the learner class
        self.epsilon = cd.run_params['epsilon']
        self.epsilon_min = cd.run_params['epsilon_min']
        self.epsilon_decay = cd.run_params['epsilon_decay']

        # Experience data
        self.buffer_capacity = cd.run_params['buffer_capacity']
        self.batch_size = cd.run_params['batch_size']

        if self.replay_buffer_type == MEMORY_TYPE.UNIFORM_REPLAY:
            self.memory = ReplayBuffer(self.buffer_capacity, self.num_states, self.num_actions)
        elif self.replay_buffer_type == MEMORY_TYPE.PRIORITY_REPLAY:
            self.memory = PrioritedReplayBuffer(self.buffer_capacity, self.num_states, self.num_actions, self.batch_size)
        elif self.replay_buffer_type == MEMORY_TYPE.HINDSIGHT_REPLAY:  # TODO: Double check if the environment has goal state
            self.memory = HindsightExperienceReplayMemory(self.buffer_capacity, self.num_states, self.num_actions)
        else:
            print("Unrecognized replay buffer please specify 'uniform, priority or hindsight', using default uniform sampling")
            raise ValueError("Unrecognized Memory type {}".format(self.replay_buffer_type))

        # Setup TF configuration to allow memory growth
        # tf.keras.backend.set_floatx('float64')
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(sess)

        # Training model only required by the learners

        if self.is_learner:
            self.actor_model = self.get_actor()
            self.critic_model_1 = self.get_critic()
            self.critic_model_2 = self.get_critic()
            self.target_actor = self.get_actor()
            self.target_critic_1 = self.get_critic()
            self.target_critic_2 = self.get_critic()
            self.target_actor.set_weights(self.actor_model.get_weights())
            self.target_critic_1.set_weights(self.critic_model_1.get_weights())
            self.target_critic_2.set_weights(self.critic_model_2.get_weights())

        # Every agent needs this, however, actors only use the CPU (for now)
        else:
            with tf.device('/CPU:0'):
                self.target_actor = self.get_actor()
                self.target_critic_1 = self.get_critic()
                self.target_critic_2 = self.get_critic()

        # Learning rate for actor-critic models
        self.critic_lr = cd.run_params['critic_lr']
        self.actor_lr = cd.run_params['actor_lr']
        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)

        self.update_actor_iter = update_actor_iter  # Updates actor every other n (2) learning rate
        self.learn_step_counter = 0
        np.random.seed(0)
        tf.random.set_seed(0)

    def remember(self, state, action, reward, next_state, done):
        # If the counter exceeds the capacity then
        self.memory.store(state, action, reward, next_state, done)

    # @tf.function
    # Just a hack for now:
    def update_grad(self, state_batch, action_batch, reward_batch, next_state_batch, terminal_batch, b_idx=None, weights=None):
        # Training and updating Actor & Critic networks.
        with tf.GradientTape(persistent=True) as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            target_actions = target_actions + tf.clip_by_value(np.random.normal(scale=0.2), -0.5, 0.5)  # TODO: Might remove this
            target_actions = tf.clip_by_value(target_actions, self.lower_bound, self.upper_bound)  # TODO: Same
            q1_ = self.target_critic_1([next_state_batch, target_actions], training=True)
            q2_ = self.target_critic_2([next_state_batch, target_actions], training=True)
            # For priroritized experience

            q1 = self.critic_model_1([state_batch, action_batch], training=True)
            q2 = self.critic_model_2([state_batch, action_batch], training=True)

            critic_value_ = tf.math.minimum(q1_, q2_)
            # print(reward_batch.shape)
            # exit()
            # reward_batch = tf.squeeze(reward_batch,1)

            y = reward_batch + self.gamma * critic_value_ * (1 - terminal_batch)

            critic_loss_1 = keras.losses.MSE(y, q1)
            critic_loss_2 = keras.losses.MSE(y, q2)
            if self.replay_buffer_type == MEMORY_TYPE.PRIORITY_REPLAY:
                error_1 = np.abs(tf.squeeze(y - q1).numpy())
                error_2 = np.abs(tf.squeeze(y - q2).numpy())
                error = np.abs(error_1 + error_2) / 2.0
                critic_loss_1 *= weights
                critic_loss_2 *= weights
                self.memory.batch_update(b_idx, error)

        logger.warning("Critic loss 1: {}, Critic loss 2: {} ".format(critic_loss_1, critic_loss_2))

        critic_grad_1 = tape.gradient(critic_loss_1, self.critic_model_1.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad_1, self.critic_model_1.trainable_variables)
        )

        critic_grad_2 = tape.gradient(critic_loss_2, self.critic_model_2.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad_2, self.critic_model_2.trainable_variables)
        )

        self.learn_step_counter += 1
        if self.learn_step_counter % self.update_actor_iter == 0:
            return

        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            critic_value_1 = self.critic_model_1([state_batch, actions], training=True)
            actor_loss = -tf.math.reduce_mean(critic_value_1)

        logger.warning("Actor loss: {}".format(actor_loss))
        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )
        # self.target_train()

    #     return model

    def get_actor(self):
        # State as input
        inputs = layers.Input(shape=(self.num_states,))
        # first layer takes inputs
        out = layers.Dense(self.actor_dense[0], activation=self.actor_dense_act)(inputs)
        # loop over remaining layers
        for i in range(1, len(self.actor_dense)):
            out = layers.Dense(self.actor_dense[i], activation=self.actor_dense_act)(out)
        # output layer has dimension actions, separate activation setting
        out = layers.Dense(self.num_actions, activation=self.actor_out_act,
                           kernel_initializer=tf.random_uniform_initializer())(out)
        outputs = layers.Lambda(lambda i: i * self.upper_bound)(out)
        model = tf.keras.Model(inputs, outputs)
        # model.summary()

        return model

    def get_critic(self):
        # State as input
        state_input = layers.Input(shape=self.num_states)
        # first layer takes inputs
        state_out = layers.Dense(self.critic_state_dense[0],
                                 activation=self.critic_state_dense_act)(state_input)
        # loop over remaining layers
        for i in range(1, len(self.critic_state_dense)):
            state_out = layers.Dense(self.critic_state_dense[i],
                                     activation=self.critic_state_dense_act)(state_out)

        # Action as input
        action_input = layers.Input(shape=self.num_actions)

        # first layer takes inputs
        action_out = layers.Dense(self.critic_action_dense[0],
                                  activation=self.critic_action_dense_act)(action_input)
        # loop over remaining layers
        for i in range(1, len(self.critic_action_dense)):
            action_out = layers.Dense(self.critic_action_dense[i],
                                      activation=self.critic_action_dense_act)(action_out)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        # assumes at least 2 post-concat layers
        # first layer takes concat layer as input
        concat_out = layers.Dense(self.critic_concat_dense[0],
                                  activation=self.critic_concat_dense_act)(concat)
        # loop over remaining inner layers
        for i in range(1, len(self.critic_concat_dense) - 1):
            concat_out = layers.Dense(self.critic_concat_dense[i],
                                      activation=self.critic_concat_dense_act)(concat_out)

        # last layer has different activation
        concat_out = layers.Dense(self.critic_concat_dense[-1], activation=self.critic_out_act,
                                  kernel_initializer=tf.random_uniform_initializer())(concat_out)
        outputs = layers.Dense(1)(concat_out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)
        # model.summary()

        return model

    def _convert_to_tensor(self, state_batch, action_batch, reward_batch, next_state_batch, terminal_batch):
        state_batch = tf.convert_to_tensor(state_batch, dtype=tf.float32)
        action_batch = tf.convert_to_tensor(action_batch, dtype=tf.float32)
        reward_batch = tf.convert_to_tensor(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(next_state_batch, dtype=tf.float32)
        terminal_batch = tf.convert_to_tensor(terminal_batch, dtype=tf.float32)
        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch

    def has_data(self):
        """Indicates if the buffer has data of size batch_size or more

        Returns:
            bool: True if replay_buffer length >= self.batch_size
        """
        return (self.memory._mem_length > 0)

    def generate_data(self):

        if self.replay_buffer_type == MEMORY_TYPE.UNIFORM_REPLAY:
            state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.memory.sample_buffer(
                self.batch_size)  # done_batch might improve experience
            state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self._convert_to_tensor(
                state_batch, action_batch, reward_batch, next_state_batch, terminal_batch)
            yield state_batch, action_batch, reward_batch, next_state_batch, terminal_batch

        elif self.replay_buffer_type == MEMORY_TYPE.PRIORITY_REPLAY:
            state_batch, action_batch, reward_batch, next_state_batch, terminal_batch, btx_idx, weights = self.memory.sample_buffer(self.batch_size)
            state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self._convert_to_tensor(
                state_batch, action_batch, reward_batch, next_state_batch, terminal_batch)
            weights = tf.convert_to_tensor(weights, dtype=tf.float32)
            yield state_batch, action_batch, reward_batch, next_state_batch, terminal_batch, btx_idx, weights
        else:
            raise ValueError('Support for the replay buffer type not implemented yet!')

    def train(self, batch):
        if self.is_learner:
            if batch and len(batch[0]) >= (self.batch_size):
                logger.warning('Training...')
                if self.replay_buffer_type == MEMORY_TYPE.UNIFORM_REPLAY:
                    self.update_grad(batch[0], batch[1], batch[2], batch[3], batch[4])
                elif self.replay_buffer_type == MEMORY_TYPE.PRIORITY_REPLAY:
                    self.update_grad(batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6])

    def target_train(self):
        # Update the target model
        model_weights = self.actor_model.get_weights()
        target_weights = self.target_actor.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = self.tau * model_weights[i] + (1 - self.tau) * target_weights[i]
        self.target_actor.set_weights(target_weights)

        model_weights = self.critic_model_1.get_weights()
        target_weights = self.target_critic_1.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = self.tau * model_weights[i] + (1 - self.tau) * target_weights[i]
        self.target_critic_1.set_weights(target_weights)

        model_weights = self.critic_model_2.get_weights()
        target_weights = self.target_critic_2.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = self.tau * model_weights[i] + (1 - self.tau) * target_weights[i]
        self.target_critic_2.set_weights(target_weights)

    def action(self, state):
        # TODO: Might be better to start after warm up
        if np.random.random() < self.epsilon:
            sampled_actions = np.random.uniform(low=self.lower_bound, high=self.upper_bound, size=(self.num_actions,))
            policy_type = 1
            self.epsilon_adj()
        else:
            policy_type = 1
            tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)

            sampled_actions = tf.squeeze(self.target_actor(tf_state))
            sampled_actions = sampled_actions.numpy()
        noise = self.ou_noise()
        sampled_actions_wn = sampled_actions + noise
        legal_action = tf.clip_by_value(sampled_actions_wn, self.lower_bound, self.upper_bound)

        return_action = [np.squeeze(legal_action)]
        logger.warning('Legal action:{}'.format(return_action))
        return return_action, policy_type

    # For distributed actors #
    def get_weights(self):
        return self.target_actor.get_weights()

    def set_weights(self, weights):
        self.target_actor.set_weights(weights)

    # Extra methods
    def update(self):
        print("Implement update method in ddpg.py")

    def load(self, file_name):
        try:
            print('... loading models ...')
            layers = self.target_actor.layers
            pickle_list = []
            for layerId in range(len(layers)):
                weigths = layers[layerId].get_weights()
                pickle_list.append([layers[layerId].name, weigths])

            with open(file_name, "wb") as f:
                pickle.dump(pickle_list, f, -1)

        except:
            # TODO: Could be improve, but ok for now
            print("One of the model not present")

    def save(self, file_name):
        try:
            print('... saving models ...')
            layers = self.target_actor.layers
            with open(file_name, "rb") as f:
                pickle_list = pickle.load(f)

            for layerId in range(len(layers)):
                assert layers[layerId].name == pickle_list[layerId][0]
                layers[layerId].set_weights(pickle_list[layerId][1])
        except:
            # TODO: Could be improve, but ok for now
            print("One of the model not present")

    def monitor(self):
        print("Implement monitor method in td3.py")

    def set_agent(self):
        print("Implement set_agent method in td3.py")

    def print_timers(self):
        print("Implement print_timers method in td3.py")

    def epsilon_adj(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
