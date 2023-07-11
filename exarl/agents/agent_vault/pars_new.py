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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from email import policy
import re
import time
import os
import math
import json
import csv
import random
from turtle import update
from typing import Type
import tensorflow as tf
import sys
import gym
import pickle
import exarl as erl
from exarl.base.comm_base import ExaComm
from tensorflow import keras
from collections import deque
from datetime import datetime
import numpy as np
from exarl.utils.introspect import introspectTrace
from tensorflow.compat.v1.keras.backend import set_session


from exarl.envs.env_vault.Hadrec_dir.src.policy_LSTM import *
from exarl.envs.env_vault.Hadrec_dir.src.utils import *
from exarl.envs.env_vault.Hadrec_dir.src.logz import *

from exarl.envs.env_vault.Hadrec_dir.src.optimizers import *

from exarl.agents.agent_vault._replay_buffer import ReplayBuffer
from exarl.utils.globals import ExaGlobals

logger = ExaGlobals.setup_logger(__name__)
import h5py

def create_shared_noise():
    """
    Create a large array of noise to be shared by all workers. Used 
    for avoiding the communication of the random perturbations delta.
    """

    seed = 12345
    count = 250000000
    noise = np.random.RandomState(seed).randn(count).astype(np.float64)
    return noise


# Code in this file is copied and adapted from
# https://github.com/ray-project/ray/blob/master/python/ray/rllib/utils/filter.py
# http://www.johndcook.com/blog/standard_deviation/
class RunningStat(object):

    def __init__(self, shape=None):
        self._n = 0
        self._M = np.zeros(shape, dtype = np.float64)
        self._S = np.ones(shape,  dtype = np.float64)

    def copy(self):
        other = RunningStat()
        other._n = self._n
        other._M = np.copy(self._M)
        other._S = np.copy(self._S)
        return other

    def push(self, x):
        x = np.asarray(x)
        # Unvectorized update of the running statistics.
        assert x.shape == self._M.shape, ("x.shape = {}, self.shape = {}"
                                          .format(x.shape, self._M.shape))
        n1 = self._n
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            delta = x - self._M
            self._M[...] += delta / self._n
            self._S[...] += delta * delta * n1 / self._n
            

    def update(self, other):
        n1 = self._n
        n2 = other._n
        n = n1 + n2
        delta = self._M - other._M
        delta2 = delta * delta
        M = (n1 * self._M + n2 * other._M) / n
        S = self._S + other._S + delta2 * n1 * n2 / n
        self._n = n
        self._M = M
        self._S = S

    def __repr__(self):
        return '(n={}, mean_mean={}, mean_std={})'.format(
            self.n, np.mean(self.mean), np.mean(self.std))

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        aa = np.array(self.var, dtype=np.float64)
        return np.sqrt(aa)
        #return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape

class SharedNoiseTable(object):
    def __init__(self, noise, seed = 11):

        self.rg = np.random.RandomState(seed)
        self.noise = noise
        assert self.noise.dtype == np.float64

    def get(self, i, dim):
        return self.noise[i:i + dim]

    def sample_index(self, dim):
        return self.rg.randint(0, len(self.noise) - dim + 1)

    def get_delta(self, dim):
        idx = self.sample_index(dim)
        return idx, self.get(idx, dim)

class PARS(erl.ExaAgent):
    def rankPrint(self, *args):
        print("AGENT_RANK:", self.agent_comm.rank, "IS_LEARNER:", self.is_learner, *args, flush=True)  
    
    """Parallel Agumented Random Search agent.
    Inherits from ExaAgent base class.
    """

    def __init__(self, env, is_learner):
        
        self.is_learner = is_learner
        self.agent_comm = ExaComm.agent_comm
        self.learner_comm = ExaComm.learner_comm
        self.epsilon = ExaGlobals.lookup_params('epsilon')

        # Get the environnemt
        self.env = env

        # self.ob_dim = env.observation_space.shape[0]
        # self.ac_dim = env.action_space.shape[0]
        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.shape[0]

        self.params = self.CreateParams()
        self.batch_size = 500

        # Define the Policy parameters
        self.policy_params = {'type': self.params['policy_type'],
                            'policy_network_size': self.params['policy_network_size'],
                            'ob_dim': self.ob_dim,
                            'ac_dim': self.ac_dim}
        
        # initialize policy
        if self.policy_params['type'] == 'linear':
            self.policy = LinearPolicy(self.policy_params)
            self.w_policy = self.policy.get_weights()
        elif self.policy_params['type'] == 'nonlinear':
            self.policy = FullyConnectedNeuralNetworkPolicy(self.policy_params, self.params['seed'])
            self.w_policy = self.policy.get_weights()
        elif self.policy_params['type'] == 'LSTM':
            self.policy = LSTMPolicy(self.policy_params)
            self.w_policy = self.policy.get_weights()													 																		
        else:
            raise NotImplementedError

        # initialize optimization algorithm
        # print('Initializing optimizer.')
        self.optimizer = SGD(self.w_policy, self.params["step_size"])
        # print("Initialization of ARS complete.")

        self.RS_deltaPerturbAllFault = RunningStat(shape=(self.ob_dim,))

        # Initialize the Running stats...
        # This for inner number of steps loop
        self.RS = RunningStat(shape=(self.ob_dim,))
        self.ob_mean = self.RS.mean
        self.ob_std = self.RS.std
        
        # These parameter are used for delta setting.
        self.step_size= self.params['step_size']
        self.delta_std = self.params['delta_std']
        self.decay = self.params['decay']
        
        # JS: Double check what this is
        # Each worker/actor picks a delta from the random stream.
        self.deltas_idx = []
        self.delta = None
        self.worker_seed_id = np.random.randint(0,high=10000)
        self.set_delta(self.worker_seed_id)

        # Collect all +ve/-ve reward means
        self.pos_neg_meanreward = []
        
        self.Num_pertub = 2  #  This is +ve and -ve perturbations direction

        # Initializat positive and negative reward
        self.pos_rew, self.neg_rew = [], []

        # This is to collect all the batches of the worker 
        # by the learner.
        self.all_actorbatch = []

        # Flag to use by set weight update and 
        self.first = True
        self.new_weights = self.Num_actors 

    def CreateParams(self):
        param = {}       
        size = ExaGlobals.lookup_params('policy_network_size')
        param['policy_network_size'] = list([size,size])
        param['delta_std'] = ExaGlobals.lookup_params('delta_std')
        param['decay'] = ExaGlobals.lookup_params('decay')
        param['rollout_length'] =  ExaGlobals.lookup_params('n_steps')
        param['seed'] = ExaGlobals.lookup_params('seed')
        param['policy_type'] = ExaGlobals.lookup_params('model_type')
        param['step_size'] = ExaGlobals.lookup_params('sgd_step_size')
        
        return param

    def set_delta(self,seed):
        deltas_id = create_shared_noise()
        self.deltas = SharedNoiseTable(deltas_id,self.params["seed"] + 7 * seed)
        idx, delta = self.deltas.get_delta(self.w_policy.size)
        self.delta = (self.delta_std * delta).reshape(self.w_policy.shape)
        self.deltas_idx.append(idx)

    def get_weights(self):
        if self.new_weights > 0:
            self.new_weights -= 1
            self.w_policy = self.policy.get_weights()
            return self.w_policy
        else:
            return None
    
    def calc_mean_pos_neg_reward(self):
        assert len(self.pos_rew) != 0 , "Positive Peturbation reward list empty"
        assert len(self.neg_rew) != 0 , "Negative Peturbation reward list empty"
        
        mean_pos_reward = np.mean(self.pos_rew)
        mean_neg_reward = np.mean(self.neg_rew)

        self.pos_neg_meanreward.append([mean_pos_reward,mean_neg_reward])
        return 
    
    def set_weights(self, weights):
        # Store the RS for all the perturb delta and faults.
        self.RS_deltaPerturbAllFault.update(self.RS)

        # Reset the mean and standard deviation based on the 
        # the run of all perturb and fault cases.
        self.ob_mean = self.RS.mean
        self.ob_std = self.RS.std

        # Reset the 
        self.RS = RunningStat(shape=(self.ob_dim,))

        # HS: Each Episode start each actor will pick a delta ...
        # This is just to make sure that each worker/actor get a 
        # unique random stream for sampling.
        # This resampling of delta for each actor is 
        # similar to the outer loop of H iteration in Alg. 1 of paper
        # ACCELERATED DERIVATIVE-FREE DEEP REINFORCEMENT LEARNING
        self.worker_seed_id = np.random.randint(0,high=10000)
        self.set_delta(self.worker_seed_id)

        self.policy.update_weights(weights)
        self.w_policy = self.policy.get_weights()
        
        # # Even count mean run with positive perturb
        # if self.internal_episode_count % 2 == 0:
        #     # update with the positive 
        #     w_pos_id = self.w_policy + self.delta
        #     self.policy.update_weights(w_pos_id)
        #     # print("Setting positive weights.. Episode:: ",self.env.workflow_episode)
        # # Odd count mean run with negative perburb 
        # else:
        #     # update with the negative perturb 
        #     w_pos_id = self.w_policy - self.delta
        #     self.policy.update_weights(w_pos_id)
        #     # print("Setting negative weights..!  Episode:: ",self.env.workflow_episode)

                   
    def action(self,state):
        ob = np.asarray(state, dtype=np.float64)

        # Calculate the normalized observation.
        normal_ob = (ob - self.ob_mean) / (self.ob_std + 1e-8)
        return self.policy.act(normal_ob), self.policy_params['type']


    def train_step(self,g_hat):
        self.rankPrint("Euclidean norm of update step:", np.linalg.norm(g_hat))
        self.w_policy -= self.optimizer._compute_step(g_hat, self.step_size).reshape(self.w_policy.shape)
        self.rankPrint('g_hat shape, w_policy shape:',np.asarray(g_hat).shape,self.w_policy.shape)

        # update the policy with new weights...
        self.policy.update_weights(self.w_policy)
        return

    def train(self,batch):
        if batch is not None:
            self.all_actorbatch.append(batch)            
            if len(self.all_actorbatch) != self.Num_actors:
                self.rankPrint("appending the batch and returning NONE")
                return None

            else:
                # use the self.all_actorbatch once all the actors have update 
                # the actorbatch list.
                assert len(self.all_actorbatch) == self.Num_actors , "one or more actors have not finished N_cases_beforeUpdate"
                
                rollout_rewards, deltas_idx, deltas_actor = [], [],[]
                for i, actorbatch in enumerate(self.all_actorbatch):
                    rollout_rewards += actorbatch['rollout_rewards']
                    deltas_idx += actorbatch['deltas_idx']
                    deltas_actor.append(actorbatch['deltas'])

                # This is the collection of all the rewards...
                rollout_rewards = np.asarray(rollout_rewards)
                max_rewards = np.max(rollout_rewards, axis=1)

                #  select top performing deltas;  95 percentile  data...
                idx = np.arange(max_rewards.size)[max_rewards >= np.percentile(max_rewards, 0.95)]
                deltas_idx = np.array(deltas_idx)[idx]
                rollout_rewards = np.array(rollout_rewards)[idx, :]
                deltas_actor = np.array(deltas_actor)[0]
                
                # normalize rewards by their standard deviation
                if np.std(rollout_rewards) > 1:
                    rollout_rewards /= np.std(rollout_rewards)

                # aggregate rollouts to form g_hat, the gradient used to compute SGD step
                g_hat, count = batched_weighted_sum(rollout_rewards[:, 0] - rollout_rewards[:, 1],
                                                        (deltas_actor.get(idx, self.w_policy.size)
                                                        for idx in deltas_idx),
                                                        batch_size=self.batch_size)
                g_hat /= deltas_idx.size

                self.step_size *= self.decay # This updated value is used at train_step 
                self.delta_std *= self.decay # This is used to update delta_std in one-worker instance.
                # This batch comes from the generate function...
                # g_hat = batch[0]
                self.train_step(g_hat)
                
                # Loss metric.
                l2_norm = np.linalg.norm(g_hat)

                self.all_actorbatch = []
                self.new_weights = self.Num_actors 
        return None 

    def generate_data(self):
        batch = {}
        # self.rankPrint(self.internal_episode_count, self.internal_episode_count, self.Num_CasesbeforeUpdate)
        # if self.internal_episode_count > 0 and self.internal_episode_count % self.Num_CasesbeforeUpdate == 0:
        if self.has_data():
            # Call the pos_neg_meanreward calc
            self.calc_mean_pos_neg_reward()

            batch['rollout_rewards'] = self.pos_neg_meanreward
            batch['deltas_idx'] = self.deltas_idx
            batch['deltas'] = self.deltas 

            # Reset the positive and negative reward list for 
            self.pos_rew, self.neg_rew = [], []
            self.pos_neg_meanreward = []
            self.deltas_idx = []

            # self.rankPrint("GenData:: >> ", batch['deltas_idx'] )
            self.rankPrint("Generate Data:: self.internal_episode_count:", self.internal_episode_count, self.env.workflow_episode, " BATCH")

            yield batch
        else:
            self.rankPrint("Generate Data:: self.internal_episode_count:", self.internal_episode_count, self.env.workflow_episode, " NONE")
            yield None

    def remember(self, state, action, reward, next_state, done):
        self.RS.push(state)
        # positive perturb policy returning
        if self.internal_episode_count % self.Num_pertub == 0:
            self.pos_rew.append(reward)
        # negative perturb policy returning
        else:
            self.neg_rew.append(reward)

    def target_train(self):
        pass

    def update_target(self):
        pass 

    def load(self):
        print("Implement load method in ARS.py")
        return 

    def save(self,fname):
        print("Implement save method in ARS.py")
        hf = h5py.File(fname, 'w')
        # Save the policy weights...
        hf.create_dataset('dataset_1', data=self.w_policy)
        hf.close()
        return 

    def set_priorities(self, indices, loss):
        pass

    def monitor(self):
        print("Implement monitor method in ARS.py")
    

    def has_data(self):
        """Indicates if the buffer has data of size batch_size or more

        Returns:
            bool: True if replay_buffer length >= self.batch_size
        """
        print("Implement has_data method in ARS.py")
        