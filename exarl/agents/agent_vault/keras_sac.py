# Copyright (c) 2020, Jefferson Science Associates, LLC. All Rights Reserved. Redistribution
# and use in source and binary forms, with or without modification, are permitted as a
# licensed user provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this
#    list of conditions and the following disclaimer in the documentation and/or other
#    materials provided with the distribution.
# 3. The name of the author may not be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# This material resulted from work developed under a United States Government Contract.
# The Government retains a paid-up, nonexclusive, irrevocable worldwide license in such
# copyrighted data to reproduce, distribute copies to the public, prepare derivative works,
# perform publicly and display publicly and to permit others to do so.
#
# THIS SOFTWARE IS PROVIDED BY JEFFERSON SCIENCE ASSOCIATES LLC "AS IS" AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL
# JEFFERSON SCIENCE ASSOCIATES, LLC OR THE U.S. GOVERNMENT BE LIABLE TO LICENSEE OR ANY
# THIRD PARTES FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import Adam

import exarl
from exarl.utils.globals import ExaGlobals
from exarl.agents.replay_buffers.replay_buffer import ReplayBuffer
logger = ExaGlobals.setup_logger(__name__)

from exarl.agents.models.tf_model import Tensorflow_Model
from copy import deepcopy

class SAC(exarl.ExaAgent):

    def __init__(self, env, is_learner, **kwargs):
        """ Define all key variables required for all agent """

        self.is_learner = is_learner
        # Get env info
        super().__init__(**kwargs)
        self.env = env

        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.upper_bound = env.action_space.high
        self.lower_bound = env.action_space.low
        print('upper_bound: ', self.upper_bound)
        print('lower_bound: ', self.lower_bound)

        # Buffer
        self.buffer_counter = 0
        self.buffer_capacity = ExaGlobals.lookup_params('buffer_capacity')
        self.batch_size = ExaGlobals.lookup_params('batch_size')
        self.memory = ReplayBuffer(self.buffer_capacity, env.observation_space, env.action_space)
        self.per_buffer = np.ones((self.buffer_capacity, 1))

        # Used to update target networks
        self.tau   = ExaGlobals.lookup_params('tau')
        self.gamma = ExaGlobals.lookup_params('gamma')
        self.alpha = ExaGlobals.lookup_params('sac_alpha')
        

        # Setup Optimizers
        critic_lr = ExaGlobals.lookup_params('critic_lr')
        actor_lr = ExaGlobals.lookup_params('actor_lr')
        self.critic_optimizer1 = Adam(critic_lr, epsilon=1e-08)
        self.critic_optimizer2 = Adam(critic_lr, epsilon=1e-08)
        self.actor_optimizer = Adam(actor_lr, epsilon=1e-08)

        self.hidden_size = 56
        self.layer_std = 1.0 / np.sqrt(float(self.hidden_size))

        # # Setup models
        self.actor  = Tensorflow_Model.create("SoftActor",
                                              observation_space=env.observation_space,
                                              action_space=env.action_space,
                                              use_gpu=self.is_learner)
        self.critic1 = Tensorflow_Model.create("SoftCritic",
                                              observation_space=env.observation_space,
                                              action_space=env.action_space,
                                              use_gpu=self.is_learner)
        self.critic2 = Tensorflow_Model.create("SoftCritic",
                                              observation_space=env.observation_space,
                                              action_space=env.action_space,
                                              use_gpu=self.is_learner)
        self.target_actor   = deepcopy(self.actor)
        self.target_critic1 = deepcopy(self.critic1)
        self.target_critic2 = deepcopy(self.critic2)

        self.actor.init_model()
        self.critic1.init_model()
        self.critic2.init_model()
        self.actor.print()
        self.critic1.print()
        self.target_actor.init_model()
        self.target_critic1.init_model()
        self.target_critic2.init_model()

        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic1.set_weights(self.critic1.get_weights())
        self.target_critic2.set_weights(self.critic2.get_weights())


        # update counting
        self.ntrain_calls = 0
        self.actor_update_freq = 2
        self.critic_update_freq = 1
        self.target_update_freq = 2

        # Not used by agent but required by the learner class
        # self.epsilon = ExaGlobals.lookup_params('epsilon')
        # self.epsilon_min = ExaGlobals.lookup_params('epsilon_min')
        # self.epsilon_decay = ExaGlobals.lookup_params('epsilon_decay')

        logger().info("TD3 buffer capacity {}".format(self.buffer_capacity))
        logger().info("TD3 batch size {}".format(self.batch_size))
        logger().info("TD3 tau {}".format(self.tau))
        logger().info("TD3 gamma {}".format(self.gamma))
        logger().info("TD3 critic_lr {}".format(critic_lr))
        logger().info("TD3 actor_lr {}".format(actor_lr))

    @tf.function
    def train_critic(self, states, actions, rewards, next_states, dones):
        # next_actions, _ = self.action(next_states, use_target=True)

        sampled_means, sampled_sds = self.actor(next_states)
        # dist         = tfp.distributions.Normal(sampled_means, sampled_sds)
        dist         = tfp.distributions.TruncatedNormal(sampled_means, sampled_sds, self.lower_bound, self.upper_bound)
        next_actions = dist.sample()
        # next_actions = tf.clip_by_value(next_actions, self.lower_bound, self.upper_bound)
        # tf.print("Means: ", sampled_means)
        # tf.print("SDs: ", sampled_sds)
        next_lp      = tf.reduce_sum(dist.log_prob(next_actions), axis=1)
        # tf.print("Next Actions: ", next_actions)
        # tf.print("Log Prob: ", next_lp)

        # Add a little noise
        new_q1 = self.target_critic1([next_states, next_actions], training=False)
        new_q2 = self.target_critic2([next_states, next_actions], training=False)
        new_q  = tf.math.minimum(new_q1, new_q2)
        # Bellman equation for the q value
        # tf.print("SHAPES: ", states.shape, actions.shape, rewards.shape, new_q.shape, next_lp.shape)
        # tf.print("Rewards: ", rewards)
        q_targets = rewards + (1.0 - dones[:,None]) * self.gamma * (new_q - self.alpha * next_lp)
        # Critic 1
        with tf.GradientTape() as tape:
            q_values1 = self.critic1([states, actions], training=True)
            td_errors1 = q_values1 - q_targets
            critic_loss1 = tf.reduce_mean(tf.math.square(td_errors1))
        # tf.print("Critic 1 Loss: ", critic_loss1)
        gradient1 = tape.gradient(critic_loss1, self.critic1.trainable_variables)
        self.critic1.optimizer.apply_gradients(zip(gradient1, self.critic1.trainable_variables))

        # Critic 2
        with tf.GradientTape() as tape:
            q_values2 = self.critic2([states, actions], training=True)
            td_errors2 = q_values2 - q_targets
            critic_loss2 = tf.reduce_mean(tf.math.square(td_errors2))
        # tf.print("Critic 2 Loss: ", critic_loss2)
        gradient2 = tape.gradient(critic_loss2, self.critic2.trainable_variables)
        self.critic2.optimizer.apply_gradients(zip(gradient2, self.critic2.trainable_variables))

    @tf.function
    def train_actor(self, states):
        # Use Critic 1
        with tf.GradientTape() as tape:
            sampled_means, sampled_sds = self.actor(states)
            dist         = tfp.distributions.TruncatedNormal(sampled_means, sampled_sds, self.lower_bound, self.upper_bound)
            # dist         = tfp.distributions.Normal(sampled_means, sampled_sds)
            actions      = dist.sample()
            # actions      = tf.clip_by_value(actions, self.lower_bound, self.upper_bound)
            action_lp    = tf.reduce_sum(dist.log_prob(actions), axis=1)
            q_value      = self.critic1([states, actions], training=True)
            loss         = -tf.math.reduce_mean(q_value - self.alpha * action_lp)
        # tf.print("Actor Loss: ", loss)
        gradient = tape.gradient(loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(gradient, self.actor.trainable_variables))

    @tf.function
    def soft_update(self, target_weights, weights):
        for (target_weight, weight) in zip(target_weights, weights):
            target_weight.assign(weight * self.tau + target_weight * (1.0 - self.tau))

    def update(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        if self.ntrain_calls % self.critic_update_freq == 0:
            self.train_critic(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
        if self.ntrain_calls % self.actor_update_freq == 0:
            self.train_actor(state_batch)

    def _convert_to_tensor(self, state_batch, action_batch, reward_batch, next_state_batch, terminal_batch):
        state_batch      = tf.convert_to_tensor(state_batch, dtype=tf.float32)
        action_batch     = tf.convert_to_tensor(action_batch, dtype=tf.float32)
        reward_batch     = tf.convert_to_tensor(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(next_state_batch, dtype=tf.float32)
        terminal_batch   = tf.convert_to_tensor(terminal_batch, dtype=tf.float32)
        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch

    def generate_data(self):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
            self._convert_to_tensor(*self.memory.sample(self.batch_size))
        yield state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def train(self, batch):
        """ Method used to train """
        self.ntrain_calls += 1
        self.update(batch[0], batch[1], batch[2], batch[3], batch[4])
        self.update_target()

    def update_target(self):
        if self.ntrain_calls % self.target_update_freq == 0:
            self.soft_update(self.target_actor.variables, self.actor.variables)
            self.soft_update(self.target_critic1.variables, self.critic1.variables)
            self.soft_update(self.target_critic2.variables, self.critic2.variables)

    def action(self, state, use_target=False):
        """ Method used to provide the next action using the target model """
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        if use_target:
            sampled_means, sampled_sds = tf.squeeze(self.target_actor(tf_state))
        else:
            sampled_means, sampled_sds = tf.squeeze(self.actor(tf_state))
        dist = tfp.distributions.TruncatedNormal(sampled_means, sampled_sds, self.lower_bound, self.upper_bound)
        # dist = tfp.distributions.Normal(sampled_means, sampled_sds)
        sampled_actions = dist.sample()

        # sampled_actions = sampled_means + sampled_sds * np.random.normal(0, 1.0, sampled_means.shape)
        policy_type = 1

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)
        # log_p        = dist.log_prob(legal_action)

        return legal_action, policy_type

    def remember(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)

    def has_data(self):
        """return true if agent has experiences from simulation
        """
        return (self.memory._mem_length > 0)

    # For distributed actors #
    def get_weights(self):
        return self.target_actor.get_weights()

    def set_weights(self, weights):
        self.target_actor.set_weights(weights)

    def train_return(self, args):
        pass


class SAC_squash(SAC):
    def __init__(self, env, is_learner, **kwargs):
        """ Define all key variables required for all agent """

        self.is_learner = is_learner
        # Get env info
        super().__init__(env, is_learner, **kwargs)

        self.target_actor  = Tensorflow_Model.create("SoftActorUnbounded",
                                              observation_space=env.observation_space,
                                              action_space=env.action_space,
                                              use_gpu=self.is_learner)
        self.actor  = Tensorflow_Model.create("SoftActorUnbounded",
                                              observation_space=env.observation_space,
                                              action_space=env.action_space,
                                              use_gpu=self.is_learner)

        self.actor.init_model()
        self.actor.print()
        self.target_actor.init_model()

        self.target_actor.set_weights(self.actor.get_weights())

    @tf.function
    def train_critic(self, states, actions, rewards, next_states, dones):

        sampled_means, sampled_sds = self.actor(next_states)
        dist         = tfp.distributions.Normal(sampled_means, sampled_sds)
        raw_actions  = dist.sample()
        next_actions = (tf.tanh(raw_actions) + 1.0) / 2.0 *(self.upper_bound - self.lower_bound) + self.lower_bound
        next_lp      = tf.reduce_sum(dist.log_prob(raw_actions),axis=1) - tf.reduce_sum(tf.math.log(1 - tf.math.square(tf.math.tanh(raw_actions))), axis=1)

        # Add a little noise
        new_q1 = self.target_critic1([next_states, next_actions], training=False)
        new_q2 = self.target_critic2([next_states, next_actions], training=False)
        new_q  = tf.math.minimum(new_q1, new_q2)
        # Bellman equation for the q value
        # tf.print("SHAPES: ", states.shape, actions.shape, rewards.shape, new_q.shape, next_lp.shape)
        # tf.print("Rewards: ", rewards)
        q_targets = rewards + (1.0 - dones[:,None]) * self.gamma * (new_q - self.alpha * next_lp)
        # Critic 1
        with tf.GradientTape() as tape:
            q_values1 = self.critic1([states, actions], training=True)
            td_errors1 = q_values1 - q_targets
            critic_loss1 = tf.reduce_mean(tf.math.square(td_errors1))
        # tf.print("Critic 1 Loss: ", critic_loss1)
        gradient1 = tape.gradient(critic_loss1, self.critic1.trainable_variables)
        self.critic1.optimizer.apply_gradients(zip(gradient1, self.critic1.trainable_variables))

        # Critic 2
        with tf.GradientTape() as tape:
            q_values2 = self.critic2([states, actions], training=True)
            td_errors2 = q_values2 - q_targets
            critic_loss2 = tf.reduce_mean(tf.math.square(td_errors2))
        # tf.print("Critic 2 Loss: ", critic_loss2)
        gradient2 = tape.gradient(critic_loss2, self.critic2.trainable_variables)
        self.critic2.optimizer.apply_gradients(zip(gradient2, self.critic2.trainable_variables))

    @tf.function
    def train_actor(self, states):
        # Use Critic 1
        with tf.GradientTape() as tape:
            sampled_means, sampled_sds = self.actor(states)
            dist         = tfp.distributions.Normal(sampled_means, sampled_sds)
            raw_actions  = dist.sample()
            actions      = (tf.tanh(raw_actions) + 1.0) / 2.0 *(self.upper_bound - self.lower_bound) + self.lower_bound
            action_lp    = tf.reduce_sum(dist.log_prob(raw_actions),axis=1) - tf.reduce_sum(tf.math.log(1 - tf.math.square(tf.math.tanh(raw_actions))), axis=1)
            q_value      = self.critic1([states, actions], training=True)
            loss         = -tf.math.reduce_mean(q_value - self.alpha * action_lp)
        # tf.print("Actor Loss: ", loss)
        gradient = tape.gradient(loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(gradient, self.actor.trainable_variables))

    def action(self, state, use_target=False):
        """ Method used to provide the next action using the target model """
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        if use_target:
            sampled_means, sampled_sds = tf.squeeze(self.target_actor(tf_state))
        else:
            sampled_means, sampled_sds = tf.squeeze(self.actor(tf_state))
        # dist = tfp.distributions.TruncatedNormal(sampled_means, sampled_sds, self.lower_bound, self.upper_bound)
        dist = tfp.distributions.Normal(sampled_means, sampled_sds)
        sampled_actions = dist.sample()
        sampled_actions = (tf.tanh(sampled_actions) + 1.0) / 2.0 *(self.upper_bound - self.lower_bound) + self.lower_bound

        policy_type = 1
        # tf.print("Means: ", sampled_means)
        # tf.print("SDs: ", sampled_sds)
        # tf.print("Actions: ", sampled_actions)

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)
        # log_p        = dist.log_prob(legal_action)

        return legal_action, policy_type
