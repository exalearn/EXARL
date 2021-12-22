import time
import os
import sys
import math
import json
import csv
import pickle
import random
import numpy as np
import tensorflow as tf
import exarl as erl
from exarl import mpi_settings
import exarl.utils.candleDriver as cd
import exarl.utils.log as log

logger = log.setup_logger(__name__, cd.run_params["log_level"])

class A2Ccontinuous(erl.ExaAgent):

    def __init__(self, env, is_learner):

        self.is_learner = False

        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.action_high = env.action_space.high[0]
        self.std_bound = [1e-2, 1.0]

        # Constants used by agent
        self.gamma = cd.run_params['gamma']
        self.e_const = cd.run_params['entropy_constant']
        self.v_const = cd.run_params['value_constant']

        self.state_memory, self.reward_memory, self.action_memory = [], [], []
        # Constants not used by agent, but needed for Learner class
        self.epsilon = cd.run_params['epsilon']
        self.epsilon_min = cd.run_params['epsilon_min']
        self.epsilon_decay = cd.run_params['epsilon_decay']

        # Actor and Critic network parameters
        self.actor_dense = cd.run_params['actor_dense']
        self.actor_dense_activation = cd.run_params['actor_dense_activation']
        self.actor_dense_kinit = cd.run_params['actor_dense_kernel_initializer']
        self.actor_lr = cd.run_params['actor_learning_rate']
        self.actor_optimizer = tf.keras.optimizers.RMSprop(self.actor_lr)

        self.critic_dense = cd.run_params['critic_dense']
        self.critic_dense_activation = cd.run_params['critic_dense_activation']
        self.critic_dense_kinit = cd.run_params['critic_dense_kernel_initializer']
        self.critic_lr = cd.run_params['critic_learning_rate']
        self.critic_optimizer = tf.keras.optimizers.RMSprop(self.critic_lr)

        # Setup TF configuration to allow memory growth
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(sess)

        # Training model is only required by the learners
        self.actor = None
        self.critic = None

        if self.is_learner:
            self.actor = Actor(self.num_actions, self.action_high, self.actor_dense, self.actor_dense_activation, self.actor_dense_kinit)
            self.critic = Critic(self.critic_dense, self.critic_dense_activation, self.critic_dense_kinit)
        else:
            with tf.device('/CPU:0'):
                self.actor = Actor(self.num_actions, self.action_high, self.actor_dense, self.actor_dense_activation, self.actor_dense_kinit)

    def action(self, state):
        mu, std = self.actor(np.array([state]))
        mu, std = mu[0], std[0]
        dist = np.random.normal(mu, std, size=self.num_actions)
        # dist = tf.compat.v1.distributions.Normal(mu, std, size=self.num_actions)
        return dist, 1

    def remember(self, state, action, reward, next_state, done):
        self.state_memory.append(state)
        self.action_memory.append(action)
        self.reward_memory.append(reward)

    def generate_data(self):
        return [self.state_memory, self.action_memory, self.reward_memory]

    def train(self, batch):
        if self.is_learner:
            logger.warning('Training...')
            self.update_grad(batch[0], batch[1], batch[2])

    def update_grad(self, state, action, reward):
        discount_rewards = []
        sum_rewards = 0.0
        reward.reverse()
        for r in reward:
            sum_rewards = r + self.gamma * sum_rewards
            discount_rewards.append(sum_rewards)

        discount_rewards.reverse()
        discount_rewards = tf.reshape(discount_rewards, (len(discount_rewards),))
        states = np.array(state, dtype=np.float32)
        actions = np.array(action, dtype=np.float32)

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            mu, std = self.actor(states, training=True)
            v = self.critic(states, training=True)
            v = tf.reshape(v, (len(v),))

            TDerror = tf.math.subtract(tf.cast(discount_rewards, tf.float32), v)
            vs = tf.math.add(v, tf.math.multiply(self.gamma, TDerror))
            vs2 = tf.math.add(TDerror, tf.math.multiply(self.gamma, vs))

            a_loss = self.actor_loss(mu, std, actions, vs2)
            c_loss = self.v_const * tf.keras.losses.mean_squared_error(vs, v)

            grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
            grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)

            self.actor_optimizer.apply_gradients(zip(grads1, self.actor.trainable_variables))
            self.critic_optimizer.apply_gradients(zip(grads2, self.critic.trainable_variables))

    def actor_loss(self, mu, std, actions, vs):

        # variance = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])**2
        # policy_pdf = tf.math.exp(-0.5*(actions-mu)**2/variance)/tf.math.sqrt(variance*2.0*np.pi)
        # log_policy_pdf = -0.5*(actions-mu)**2/variance - 0.5*tf.math.log(variance*2.0*np.pi)
        #
        # policy_pdf = tf.reduce_sum(policy_pdf, 1, keepdims=True)
        # log_policy_pdf = tf.reduce_sum(log_policy_pdf, 1, keepdims=True)
        #
        # policy_loss = tf.math.multiply(log_policy_pdf, vs)
        # entropy_loss = tf.math.negative(tf.math.multiply(policy_pdf, log_policy_pdf))

        # return tf.reduce_sum(-policy_loss)-self.e_const*tf.reduce_sum(entropy_loss)
        # return tf.reduce_sum(-policy_loss)

        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        normal_dist = tf.compat.v1.distributions.Normal(mu, std)
        log_prob = normal_dist.log_prob(actions)
        policy_loss = log_prob * vs
        entropy = normal_dist.entropy()
        policy_loss = tf.reduce_sum(self.e_const * entropy + policy_loss)
        return tf.negative(policy_loss)

    def target_train(self):
        pass

    def reset_lists(self):
        self.state_memory, self.reward_memory, self.action_memory = [], [], []

    def set_learner(self):
        self.is_learner = True
        self.actor = Actor(self.num_actions, self.action_high, self.actor_dense, self.actor_dense_activation, self.actor_dense_kinit)
        self.critic = Critic(self.critic_dense, self.critic_dense_activation, self.critic_dense_kinit)

    def get_weights(self):
        return self.actor.get_weights()

    def set_weights(self, weights):
        self.actor.set_weights(weights)

    def load(self, filename):
        layers = self.actor.layers
        with open(filename, "rb") as f:
            pickle_list = pickle.load(f)

        for layerId in range(len(layers)):
            assert layers[layerId].name == pickle_list[layerId][0]
            layers[layerId].set_weights(pickle_list[layerId][1])

    def save(self, filename):
        layers = self.actor.layers
        pickle_list = []
        for layerId in range(len(layers)):
            weigths = layers[layerId].get_weights()
            pickle_list.append([layers[layerId].name, weigths])

        with open(filename, "wb") as f:
            pickle.dump(pickle_list, f, -1)

    def epsilon_adj(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class Actor(tf.keras.Model):

    def __init__(self, nactions, act_high, ndense, act, kinit):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(ndense, activation=act, kernel_initializer=kinit)
        self.d2 = tf.keras.layers.Dense(ndense, activation=act, kernel_initializer=kinit)
        self.mu = tf.keras.layers.Dense(nactions, activation='tanh')
        self.mu_output = tf.keras.layers.Lambda(lambda x: x * act_high)
        self.std_output = tf.keras.layers.Dense(nactions, activation='softplus')

    def call(self, input_data):
        x1 = self.d1(input_data)
        x1 = self.d2(x1)
        x2 = self.mu(x1)
        mu_out = self.mu_output(x2)
        std_out = self.std_output(x1)
        return [mu_out, std_out]

class Critic(tf.keras.Model):

    def __init__(self, ndense, act, kinit):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(ndense, activation=act, kernel_initializer=kinit)
        self.d2 = tf.keras.layers.Dense(ndense, activation=act, kernel_initializer=kinit)
        self.d3 = tf.keras.layers.Dense(ndense, activation=act, kernel_initializer=kinit)
        self.v = tf.keras.layers.Dense(1, activation=None)

    def call(self, input_data):
        x = self.d1(input_data)
        x = self.d2(x)
        x = self.d3(x)
        return self.v(x)
