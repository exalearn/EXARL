import time
import os
import math
import json
import csv
import random
import tensorflow as tf
import sys
import pickle
import exarl as erl
from exarl import mpi_settings
from datetime import datetime
import numpy as np
import exarl.utils.candleDriver as cd
import exarl.utils.log as log

logger = log.setup_logger(__name__, cd.run_params["log_level"])

class A2C(erl.ExaAgent):

    def __init__(self, env, is_learner):

        self.is_learner = False

        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.n

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

        # Training model only required by the learners
        self.actor = None
        self.critic = None

        if self.is_learner:
            self.actor = Actor(self.num_actions, self.actor_dense, self.actor_dense_activation, self.actor_dense_kinit)
            self.critic = Critic(self.critic_dense, self.critic_dense_activation, self.critic_dense_kinit)
        else:
            with tf.device('/CPU:0'):
                self.actor = Actor(self.num_actions, self.actor_dense, self.actor_dense_activation, self.actor_dense_kinit)

    def action(self, state):
        prob = self.actor(np.array([state]))
        prob = prob.numpy()
        dist = tf.compat.v1.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0]), 1

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
            p = self.actor(states, training=True)
            v = self.critic(states, training=True)
            v = tf.reshape(v, (len(v),))
            TDerror = tf.math.subtract(discount_rewards, v)

            a_loss = self.actor_loss(p, actions, TDerror)
            c_loss = self.v_const * tf.keras.losses.mean_squared_error(discount_rewards, v)

            grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
            grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)

            self.actor_optimizer.apply_gradients(zip(grads1, self.actor.trainable_variables))
            self.critic_optimizer.apply_gradients(zip(grads2, self.critic.trainable_variables))

    def actor_loss(self, probs, actions, TDerror):
        probability = []
        log_probability = []

        for pb, act in zip(probs, actions):
            dist = tf.compat.v1.distributions.Categorical(probs=pb, dtype=tf.float32)
            probability.append(dist.prob(act))
            log_probability.append(dist.log_prob(act))

        p_loss = []
        e_loss = []
        TDerror = TDerror.numpy()

        for pb, t, lpb in zip(probability, TDerror, log_probability):
            t = tf.constant(t)
            policy_loss = tf.math.multiply(lpb, t)
            entropy_loss = tf.math.negative(tf.math.multiply(pb, lpb))
            p_loss.append(policy_loss)
            e_loss.append(entropy_loss)

        p_loss = tf.reduce_mean(tf.stack(p_loss))
        e_loss = tf.reduce_mean(tf.stack(e_loss))

        return -p_loss - self.e_const * e_loss

    def target_train(self):
        pass

    def reset_lists(self):
        self.state_memory, self.reward_memory, self.action_memory = [], [], []

    def set_learner(self):
        self.is_learner = True
        self.actor = Actor(self.num_actions, self.actor_dense, self.actor_dense_activation, self.actor_dense_kinit)
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

    def __init__(self, nactions, ndense, act, kinit):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(ndense, activation=act, kernel_initializer=kinit)
        self.a = tf.keras.layers.Dense(nactions, activation='softmax')

    def call(self, input_data):
        x = self.d1(input_data)
        return self.a(x)

class Critic(tf.keras.Model):

    def __init__(self, ndense, act, kinit):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(ndense, activation=act, kernel_initializer=kinit)
        self.v = tf.keras.layers.Dense(1, activation=None)

    def call(self, input_data):
        x = self.d1(input_data)
        return self.v(x)
