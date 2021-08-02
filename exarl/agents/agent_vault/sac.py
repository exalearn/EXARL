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
import tensorflow_probability as tfp
import random
import os
import sys
import pickle
from datetime import datetime
from exarl.utils.OUActionNoise import OUActionNoise
from exarl.utils.OUActionNoise import OUActionNoise2
#from ._network_sac import CriticModel, ValueModel, ActorModel
from exarl.utils.memory_type import MEMORY_TYPE
from ._replay_buffer import ReplayBuffer, HindsightExperienceReplayMemory, PrioritedReplayBuffer

import exarl as erl

import exarl.utils.log as log
import exarl.utils.candleDriver as cd
logger = log.setup_logger(__name__, cd.run_params['log_level'])


class SAC(erl.ExaAgent):

    def __init__(self, env, is_learner=False,scale=10):
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
        self.repram = cd.run_params['repram']
        

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

        self.value_state_dense = cd.run_params['value_state_dense']
        self.value_state_dense_act = cd.run_params['value_state_dense_act']
        self.value_action_dense = cd.run_params['value_action_dense']
        self.value_action_dense_act = cd.run_params['value_action_dense_act']
        self.value_concat_dense = cd.run_params['value_concat_dense']
        self.value_concat_dense_act = cd.run_params['value_concat_dense_act']
        self.value_out_act = cd.run_params['value_out_act']
        self.value_optimizer = cd.run_params['value_optimizer']

        self.replay_buffer_type = cd.run_params['replay_buffer_type']
        #self.directory = cd.run_params["output_dir"]
        #print(self.actor_dense, self.actor_dense_act, self.actor_out_act,self.value_dense,self.value_dense_act,self.critic_out_act )
        
        # Noise for SAC model
        std_dev = cd.run_params['std_dev']
        ave_bound = (self.upper_bound + self.lower_bound) / 2
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
        elif self.replay_buffer_type == MEMORY_TYPE.HINDSIGHT_REPLAY: # TODO: Double check if the environment has goal state
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

        if self.is_learner:
            self.critic_model_1 = self.get_critic()
            self.critic_model_2 = self.get_critic()
            self.value_model = self.get_value()
            self.actor_model = self.get_actor()
            self.target_value = self.get_value()
            self.target_value.set_weights(self.value_model.get_weights())
            
        else:
            with tf.device('/CPU:0'):
                self.target_value = self.get_value()
                self.actor_model = self.get_actor()
        
        self.critic_lr = cd.run_params['critic_lr']
        self.actor_lr = cd.run_params['actor_lr']
        self.value_lr = cd.run_params['value_lr']
        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)
        self.value_optimizer = tf.keras.optimizers.Adam(self.value_lr)


    def remember(self, state, action, reward, next_state, done):
        # If the counter exceeds the capacity then
        self.memory.store(state, action, reward, next_state, done)

    def update_grad(self, state_batch, action_batch, reward_batch, next_state_batch, terminal_batch,b_idx=None,weights=None):

        with tf.GradientTape() as tape:
            value = self.value_model(state_batch, training=True)
            #print(value)
            value_next = self.target_value(next_state_batch, training=True)
            policy_actions, log_probs = self.sample_normal(state_batch, reparameterize=False)
            #log_probs = tf.squeeze(log_probs, 1)
            q1_new_policy = self.critic_model_1([state_batch, policy_actions], training=True)
            q2_new_policy = self.critic_model_2([state_batch, policy_actions], training=True)
            critic_value = tf.math.minimum(q1_new_policy, q2_new_policy)
            value_target = critic_value - log_probs
            value_loss = 0.5 * keras.losses.MSE(value, value_target)
        logger.warning("Value loss: {}".format(value_loss))
        value_grad = tape.gradient(value_loss, self.value_model.trainable_variables)
        self.value_optimizer.apply_gradients(
            zip(value_grad, self.value_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            new_policy_actions , log_probs = self.sample_normal(state_batch, reparameterize=True)
            #log_probs = tf.squeeze(log_probs, 1)
            q1_new_policy = self.critic_model_1([state_batch, new_policy_actions], training=True)
            q2_new_policy = self.critic_model_2([state_batch, new_policy_actions], training=True)
            critic_value = tf.math.minimum(q1_new_policy, q2_new_policy)
            actor_target = log_probs - critic_value
            actor_loss = tf.math.reduce_mean(actor_target)
        logger.warning("Actor loss: {}".format(actor_loss))
        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )

        with tf.GradientTape(persistent=True) as tape:
            q_hat = self.scale*reward_batch + self.gamma*value_next*(1-terminal_batch)
            q1_old_policy = self.critic_model_1([state_batch, action_batch], training=True)
            q2_old_policy = self.critic_model_2([state_batch, action_batch], training=True)
            critic_1_loss = 0.5 * keras.losses.MSE(q1_old_policy, q_hat)
            critic_2_loss = 0.5 * keras.losses.MSE(q2_old_policy, q_hat)
            #print(critic_1_loss,critic_2_loss)
            
            if self.replay_buffer_type == MEMORY_TYPE.PRIORITY_REPLAY:
                error_1 = tf.squeeze(q_hat - q1_old_policy).numpy()
                error_2 = tf.squeeze(q_hat - q2_old_policy).numpy()
                error = np.abs(error_1 + error_2)/2.0
                print(critic_1_loss, critic_2_loss)
                critic_1_loss *= weights
                critic_2_loss *= weights
                self.memory.batch_update(b_idx, error)

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
        self.target_train()


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
        mu = layers.Lambda(lambda i: i * self.upper_bound)(out) # For mu
        sigma = layers.Lambda(lambda i: i * self.upper_bound)(out) # For sigma
        model = tf.keras.Model(inputs, [mu, sigma])
        #model.summary()
        #exit()
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
        #model.summary()

        return model

    def get_value(self):
        # State as input
        state_input = layers.Input(shape=self.num_states)
        # first layer takes inputs
        state_out = layers.Dense(self.value_state_dense[0],
                                 activation=self.value_state_dense_act)(state_input)
        # loop over remaining layers
        for i in range(1, len(self.value_state_dense)):
            state_out = layers.Dense(self.value_state_dense[i],
                                     activation=self.value_state_dense_act)(state_out)

        # assumes at least 2 post-concat layers
        # first layer takes concat layer as input
        concat_out = layers.Dense(self.value_concat_dense[0],
                                  activation=self.value_concat_dense_act)(state_out)
        # loop over remaining inner layers
        for i in range(1, len(self.value_concat_dense) - 1):
            concat_out = layers.Dense(self.value_concat_dense[i],
                                      activation=self.value_concat_dense_act)(concat_out)

        # last layer has different activation
        concat_out = layers.Dense(self.value_concat_dense[-1], activation=self.value_out_act,
                                  kernel_initializer=tf.random_uniform_initializer())(concat_out)
        outputs = layers.Dense(1)(concat_out)

        # Outputs single value for give state-action
        model = tf.keras.Model(state_input, outputs)
        #model.summary()
        return model

    def sample_normal(self, state_batch, reparameterize=True):

        mu, sigma = self.actor_model(state_batch, training=True)

        sigma = tf.clip_by_value(sigma, self.repram, 1)
        probabilities = tfp.distributions.Normal(mu, sigma)
        if reparameterize:
            actions = probabilities.sample() # TODO: add reparameterization here, maybe this will improve accuracy
        else:
            actions = probabilities.sample()
        
        action = tf.math.tanh(actions)*self.upper_bound
        #action = tf.math.scalar_mul(tf.constant(self.upper_bound, dtype=tf.float32),tf.math.tanh(actions))
        log_probs = probabilities.log_prob(actions)
        log_probs -= tf.math.log(1-tf.math.pow(action, 2) + self.repram) #noise to avoid taking log of zero
        #log_probs -= tf.math.log(1-tf.math.pow(action, 2) + self.ou_noise)
        log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)
        print(action[0], log_probs[0])

        return action, log_probs

    def _convert_to_tensor(self, state_batch, action_batch, reward_batch, next_state_batch, terminal_batch):
        state_batch = tf.convert_to_tensor(state_batch,dtype=tf.float32)
        action_batch = tf.convert_to_tensor(action_batch,dtype=tf.float32)
        reward_batch = tf.convert_to_tensor(reward_batch,dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(next_state_batch,dtype=tf.float32)
        terminal_batch = tf.convert_to_tensor(terminal_batch,dtype=tf.float32)
        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch

    def generate_data(self):
        #TODO: Think of a better way to do this.
        if self.replay_buffer_type == MEMORY_TYPE.UNIFORM_REPLAY:
            state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.memory.sample_buffer(self.batch_size) #done_batch might improve experience
            state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self._convert_to_tensor(state_batch, action_batch, reward_batch, next_state_batch, terminal_batch)
            yield state_batch, action_batch, reward_batch, next_state_batch, terminal_batch

        elif self.replay_buffer_type == MEMORY_TYPE.PRIORITY_REPLAY:
            state_batch, action_batch, reward_batch, next_state_batch, terminal_batch , btx_idx ,weights = self.memory.sample_buffer(self.batch_size)

            state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self._convert_to_tensor(state_batch, action_batch, reward_batch, next_state_batch,terminal_batch)
            weights = tf.convert_to_tensor(weights,dtype=tf.float32)
            yield state_batch, action_batch, reward_batch, next_state_batch, terminal_batch, btx_idx, weights
        else:
            raise ValueError('Support for the replay buffer type not implemented yet!')
    #TODO: Replace alot of if-else statement with switch statement
    def train(self, batch):
        if self.is_learner:
            if batch and len(batch[0]) >= (self.batch_size):
                logger.warning('Training...')
                if self.replay_buffer_type == MEMORY_TYPE.UNIFORM_REPLAY:
                    self.update_grad(batch[0], batch[1], batch[2], batch[3],batch[4])
                elif self.replay_buffer_type == MEMORY_TYPE.PRIORITY_REPLAY:
                    self.update_grad(batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6])
                else:
                    raise ValueError('Support for the replay buffer type not implemented yet!')
        
    def target_train(self):
        model_weights = self.value_model.get_weights()
        target_weights = self.target_value.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = self.tau * model_weights[i] + (1 - self.tau)* target_weights[i]
        self.target_value.set_weights(target_weights)
    
    def action(self, state):
        policy_type = 1
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)

        sampled_actions, _ = tf.squeeze(self.sample_normal(tf_state, reparameterize=False))
        sampled_actions_wn = sampled_actions.numpy()
        isValid = self.env.action_space.contains(sampled_actions_wn)
        if isValid == False:
            legal_action = np.random.uniform(low=self.lower_bound, high=self.upper_bound, size=(self.num_actions,))
            policy_type = 0
            logger.warning('Bad action: {}; Replaced with: {}'.format(sampled_actions_wn, legal_action))
            #logger.warning('Policy action: {}; noise: {}'.format(sampled_actions, noise))

        return_action = [np.squeeze(legal_action)]
        logger.warning('Legal action:{}'.format(return_action))
        return return_action, policy_type

    # For distributed actors #    
    def get_weights(self):
        return self.target_value.get_weights()

    def set_weights(self, weights):
        self.target_value.set_weights(weights)

    def set_learner(self):
        self.is_learner = True
        self.critic_model_1 = self.get_critic()
        self.critic_model_2 = self.get_critic()
        self.value_model = self.get_value()
        self.actor_model = self.get_actor()
        self.target_value = self.get_value()
        self.target_value.set_weights(self.value_model.get_weights())

    def update(self):
        print("Implement update method in sac.py")

    def load(self, file_name):
        #TODO: Sanity check to verify the model is there
        try:
            print("... loading Models ...")
            self.actor_model.load_weights(file_name)
            self.critic_model_1.load_weights(file_name)
            self.critic_model_2.load_weights(file_name)
            self.value_model.load_weights(file_name)
            self.target_value.load_weights(file_name)
        except:
            #TODO: Could be improved, but ok for now
            print("One of the model not present")

    def save(self, file_name):

        try:
            print("... Saving Models ...")
            self.actor_model.save_weights(file_name)
            self.critic_model_1.save_weights(file_name)
            self.critic_model_2.save_weights(file_name)
            self.value_model.save_weights(file_name)
            self.target_value.save_weights(file_name)
        except:
            #TODO: Could be improved, but ok for now
            print("One of the model not present")

    def monitor(self):
        print("Implement monitor method in sac.py")

    def set_agent(self):
        print("Implement set_agent method in sac.py")

    def print_timers(self):
        print("Implement print_timers method in sac.py")

    def epsilon_adj(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
