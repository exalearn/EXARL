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
import random
import os
import sys
import pickle
from datetime import datetime
from exarl.utils.OUActionNoise import OUActionNoise
from exarl.utils.OUActionNoise import OUActionNoise2
from exarl.agents.agent_vault.network import CriticModel, ValueModel, ActorModel

import exarl as erl

import exarl.utils.log as log
import exarl.utils.candleDriver as cd
logger = log.setup_logger(__name__, cd.run_params['log_level'])


class SAC(erl.ExaAgent):

    def __init__(self, env, is_learner=False,scale=2):
        self.is_learner = is_learner
        self.env = env
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.upper_bound = env.action_space.high
        self.lower_bound = env.action_space.low
        self.scale = scale

        logger.info("Size of State Space:  {}".format(self.num_states))
        logger.info("Size of Action Space:  {}".format(self.num_actions))
        logger.info('Env upper bounds: {}'.format(self.upper_bound))
        logger.info('Env lower bounds: {}'.format(self.lower_bound))

        self.gamma = cd.run_params['gamma']
        self.tau = cd.run_params['tau']

        # model definitions
        #TODO: Using the same parameters for critic 1 and 2, might change later
        self.actor_dense = cd.run_params['actor_dense']
        self.actor_dense_act = cd.run_params['actor_dense_act']
        self.actor_out_act = cd.run_params['actor_out_act']
        #self.actor_optimizer = cd.run_params['actor_optimizer']
        self.value_dense = cd.run_params['value_dense']
        self.value_dense_act = cd.run_params['value_dense_act']
        self.value_out_act = cd.run_params['value_out_act']
        self.value_optimizer = cd.run_params['value_optimizer']
        self.critic_dense = cd.run_params['critic_dense']
        self.critic_dense_act = cd.run_params['critic_dense_act']
        self.critic_out_act = cd.run_params['critic_out_act']
        #self.critic_optimizer = cd.run_params['critic_optimizer']
        self.directory = cd.run_params["output_dir"]

        # Noise for SAC model
        std_dev = cd.run_params['std_dev']
        ave_bound = (self.upper_bound + self.lower_bound) / 2
        print('ave_bound: {}'.format(ave_bound))
        self.ou_noise = OUActionNoise(mean=ave_bound, std_deviation=float(std_dev) * np.ones(1))
        print(self.ou_noise)
        
        # Not used by agent but required by the learner class
        self.epsilon = cd.run_params['epsilon']
        self.epsilon_min = cd.run_params['epsilon_min']
        self.epsilon_decay = cd.run_params['epsilon_decay']

        # Experience data
        self.buffer_counter = 0
        self.buffer_capacity = cd.run_params['buffer_capacity']
        self.batch_size = cd.run_params['batch_size']

        self.state_buffer = np.zeros((self.buffer_capacity, self.num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, self.num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, self.num_states))
        self.done_buffer = np.zeros((self.buffer_capacity, 1))
        self.memory =  np.zeros((self.buffer_capacity, self.num_states))  # TODO: Change this

        # Setup TF configuration to allow memory growth
        # tf.keras.backend.set_floatx('float64')
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(sess)

        
        self.critic_model_1 = None
        self.critic_model_2 = None
        self.value_model = None
        self.actor_model = None
        if self.is_learner:
            self.critic_model_1 = self.get_critic(name='Critic_1')
            self.critic_model_2 = self.get_critic(name='Critic_2')
            self.value_model = self.get_value()
            self.actor_model = self.get_actor()
        else:
            self.actor_model = self.get_actor()

        self.target_value = None
        if self.is_learner:
            self.target_value = self.get_value(name='target_value')
            self.target_value.set_weights(self.value_model.get_weights())
            
        else:
            with tf.device('/CPU:0'):
                self.target_value = self.get_value(name='target_value')

        self.critic_lr = cd.run_params['critic_lr']
        self.actor_lr = cd.run_params['actor_lr']
        self.value_lr = cd.run_params['value_lr']
        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)
        self.value_optimizer = tf.keras.optimizers.Adam(self.value_lr)
        #self.actor_model.compile(tf.keras.optimizers.Adam(self.actor_lr))
        #self.critic_model_1.compile(tf.keras.optimizers.Adam(self.critic_lr))
        #self.critic_model_2.compile(tf.keras.optimizers.Adam(self.critic_lr))
        #self.value_model.compile(tf.keras.optimizers.Adam(self.value_lr))
        #self.target_value.compile(tf.keras.optimizers.Adam(self.value_lr))

    def remember(self, state, action, reward, next_state, done):
        # If the counter exceeds the capacity then
        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = state
        self.action_buffer[index] = action[0]
        self.reward_buffer[index] = reward
        self.next_state_buffer[index] = next_state
        self.done_buffer[index] = int(done)
        self.buffer_counter += 1

    def update_grad(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):

        with tf.GradientTape() as tape:

            value = tf.squeeze(self.value_model(state_batch), 1)
            value_next = tf.squeeze(self.target_value(next_state_batch), 1)
            policy_actions, log_probs = self.actor_model.sample_normal(state_batch, reparameterize=False)
            log_probs = tf.squeeze(log_probs, 1)
            q1_new_policy = self.critic_model_1(state_batch, policy_actions)
            q2_new_policy = self.critic_model_2(state_batch, policy_actions)
            critic_value = tf.squeeze(tf.math.minimum(q1_new_policy, q2_new_policy), 1)
            value_target = critic_value - log_probs
            value_loss = 0.5 * keras.losses.MSE(value, value_target)
        logger.warning("Value loss: {}".format(value_loss))
        value_grad = tape.gradient(value_loss, self.value_model.trainable_variables)
        self.value_optimizer.apply_gradients(
            zip(value_grad, self.value_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            new_policy_actions , log_probs = self.actor_model.sample_normal(state_batch, reparameterize=True)
            log_probs = tf.squeeze(log_probs, 1)
            q1_new_policy = self.critic_model_1(state_batch, new_policy_actions)
            q2_new_policy = self.critic_model_2(state_batch, new_policy_actions)
            critic_value = tf.squeeze(tf.math.minimum(q1_new_policy, q2_new_policy), 1)
            actor_target = log_probs - critic_value
            actor_loss = tf.math.reduce_mean(actor_target)
        logger.warning("Actor loss: {}".format(actor_loss))
        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )

        with tf.GradientTape(persistent=True) as tape:
            q_hat = self.scale*reward_batch + self.gamma*value_next*(1-done_batch)
            q1_old_policy = tf.squeeze(self.critic_model_1(state_batch, action_batch), 1)
            q2_old_policy = tf.squeeze(self.critic_model_2(state_batch, action_batch), 1)
            critic_1_loss = 0.5 * keras.losses.MSE(q1_old_policy, q_hat)
            critic_2_loss = 0.5 * keras.losses.MSE(q2_old_policy, q_hat)

        logger.warning("Critic 1 loss: {}".format(critic_1_loss))
        logger.warning("Critic 2 loss: {}".format(critic_2_loss))

        critic_1_grad = tape.gradient(critic_1_loss, self.critic_model_1.trainable_variables)
        critic_2_grad = tape.gradient(critic_2_loss, self.critic_model_2.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_1_grad, self.critic_model_1.trainable_variables)
        )
        self.critic_optimizer.apply_gradients(
            zip(critic_2_grad, self.critic_model_2.trainable_variables)
        )


    def generate_data(self):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        logger.info('record_range:{}'.format(record_range))
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)
        logger.info('batch_indices:{}'.format(batch_indices))
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices], dtype=tf.float32)
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices], dtype=tf.float32)
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices], dtype=tf.float32)
        #reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices], dtype=tf.float32)

        yield state_batch, action_batch, reward_batch, next_state_batch

    def get_actor(self, name='Actor'):
        model = ActorModel(self.upper_bound, self.actor_dense, self.num_actions, name, self.directory, self.actor_dense_act,self.actor_out_act) #self.ou_noise
        return model

    def get_critic(self, name='Critic'):
        model = CriticModel(self.num_actions, self.critic_dense,name, self.directory, self.critic_dense_act,self.critic_out_act)
        return model

    def get_value(self, name='Value'):
        model = ValueModel( self.value_dense,name, self.directory, self.value_dense_act,self.value_out_act)
        return model

    def action(self, state):
        policy_type = 1
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)

        sampled_actions, _ = tf.squeeze(self.actor_model.sample_normal(tf_state, reparameterize=False))
        #noise = self.ou_noise()
        sampled_actions_wn = sampled_actions.numpy()
        legal_action = sampled_actions_wn
        isValid = self.env.action_space.contains(sampled_actions_wn)
        if isValid == False:
            legal_action = np.random.uniform(low=self.lower_bound, high=self.upper_bound, size=(self.num_actions,))
            policy_type = 0
            logger.warning('Bad action: {}; Replaced with: {}'.format(sampled_actions_wn, legal_action))
            #logger.warning('Policy action: {}; noise: {}'.format(sampled_actions, noise))

        return_action = [np.squeeze(legal_action)]
        logger.warning('Legal action:{}'.format(return_action))
        return return_action, policy_type

    def train(self, batch, done_batch):
        if self.is_learner:
            logger.warning('Training...')
            self.update_grad(batch[0], batch[1], batch[2], batch[3],done_batch)

    def target_train(self):
        model_weights = self.value_model.get_weights()
        target_weights = self.target_value.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = self.tau * model_weights[i] + (1 - self.tau)* target_weights[i]
        self.target_value.set_weights(target_weights)


    # For distributed actors #
    
    def get_weights(self):
        return self.target_value.get_weights()

    def set_weights(self, weights):
        self.target_value.set_weights(weights)

    def set_learner(self):
        self.is_learner = True
        self.critic_model_1 = self.get_critic(name='Critic_1')
        self.critic_model_2 = self.get_critic(name='Critic_2')
        self.value_model = self.get_value()
        self.actor_model = self.get_actor()
        self.target_value = self.get_value(name='target_value')
        self.target_value.set_weights(self.value_model.get_weights())

    def update(self):
        print("Implement update method in sac.py")

    def load(self, filename):
        #TODO: Sanity check to verify the model is there
        print("... loading Models ...")
        self.actor_model.load_weights(self.actor_model.get_checkpoint_name())
        self.critic_model_1.load_weights(self.critic_model_1.get_checkpoint_name())
        self.critic_model_2.load_weights(self.critic_model_2.get_checkpoint_name())
        self.value_model.load_weights(self.value_model.get_checkpoint_name())
        self.target_model.load_weights(self.target_model.get_checkpoint_name())

    def save(self, dir_name=None):
        print("... Saving Models ...")
        self.actor_model.save_weights(self.actor_model.get_checkpoint_name())
        self.critic_model_1.save_weights(self.critic_model_1.get_checkpoint_name())
        self.critic_model_2.save_weights(self.critic_model_2.get_checkpoint_name())
        self.value_model.save_weights(self.value_model.get_checkpoint_name())
        self.target_model.save_weights(self.target_model.get_checkpoint_name())

    def monitor(self):
        print("Implement monitor method in sac.py")

    def set_agent(self):
        print("Implement set_agent method in sac.py")

    def print_timers(self):
        print("Implement print_timers method in sac.py")

    def epsilon_adj(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
