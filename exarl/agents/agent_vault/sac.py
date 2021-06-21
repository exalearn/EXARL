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
from tensorflow.keras import layers
import random
import os
import pickle
from datetime import datetime

import exarl as erl

import exarl.utils.log as log
import exarl.utils.candleDriver as cd
logger = log.setup_logger(__name__, cd.run_params['log_level'])

@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

class SAC(erl.ExaAgent):

    def __init__(self, env, is_learner):
        self.is_learner = False
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
        self.tau = cd.run_params['tau']


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

        