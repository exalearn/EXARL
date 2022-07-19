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



class SingleRolloutSlaver(object):
    '''
    Responsible for doing one single rollout in the env
    '''
    def __init__(self, env,
                    rollout_length, 
                    policy_params,
                    OB_DIM=None):

        self.env = env
        self.OB_DIM=OB_DIM
        self.rollout_length = rollout_length
        self.policy_type = policy_params['type']
        
        if policy_params['type'] == 'linear':
            self.policy = LinearPolicy(policy_params)
        elif policy_params['type'] == 'nonlinear':
            self.policy = FullyConnectedNeuralNetworkPolicy(policy_params)
        elif policy_params['type'] == 'LSTM':
            self.policy = LSTMPolicy(policy_params)
        else:
            raise NotImplementedError
        
    def single_rollout(self, fault_tuple, weights, ob_mean, ob_std):
        # one SingleRolloutSlaver is only doing one fault case in an iter
        # fault_tuple = PF_FAULT_CASES_ALL[fault_case_id]
        total_reward = 0.
        steps = 0												
        self.policy.update_weights(weights)
        
        # us RS for collection of observation states; restart every rollout
        self.RS = RunningStat(shape=(self.OB_DIM,))

        t1 = time.time()											  
        ob = self.env.validate(case_Idx=fault_tuple[0], fault_bus_idx=fault_tuple[1],
                               fault_start_time=fault_tuple[2], fault_duration_time=fault_tuple[3])

        if self.policy_type == 'LSTM':
            self.policy.reset()


        t_act = 0
        t_step = 0
        for _ in range(self.rollout_length):
            ob = np.asarray(ob, dtype=np.float64)
            self.RS.push(ob)
            normal_ob = (ob - ob_mean) / (ob_std + 1e-8)
            t3 = time.time()
            action_org = self.policy.act(normal_ob)
            t4 = time.time()
            ob, reward, done, _ = self.env.step(action_org)           
            t5 = time.time()
            t_act += t4 - t3
            t_step += t5 - t4            
            total_reward += reward
            steps += 1														   
            if done:
                break
        t2 = time.time()					 

        # if b_debug_timing:
        #     return {'reward': total_reward, 'step': steps,'time': [t2-t1 , t_act, t_step, t1, t2]}
        # else:
        return {'reward': total_reward, 'step': steps}

    def return_RS(self):
        return self.RS

    def close_env(self):
        self.env.close_env()
        return





class OneDirectionWorker(object):
    """
    Object class for parallel rollout generation.
    """

    def __init__(self, env, 
                 env_seed,
                 policy_params=None,
                 deltas=None,
                 rollout_length=None,
                 delta_std=None,
                 PF_FAULT_CASES_ALL=None,
                 OB_DIM=None):

        # each worker gets access to the shared noise table
        # with independent random streams for sampling
        # from the shared noise table.
        self.deltas = SharedNoiseTable(deltas, env_seed + 7)
        self.policy_params = policy_params
        self.delta_std = delta_std
        self.rollout_length = rollout_length
        self.OB_DIM = OB_DIM

        self.num_slaves = len(PF_FAULT_CASES_ALL)
        
        # explore positive and negative delta at the same time for parallel execution
        self.single_rollout_slavers = []

        # TODO: The ExaRL should try to parallelize this for loop...
        # Parallelize this step...
        for i in range(self.num_slaves):
            self.single_rollout_slavers.append(SingleRolloutSlaver(env,rollout_length=rollout_length,
                                            policy_params=self.policy_params,
                                            OB_DIM=self.OB_DIM))
            

    def update_delta_std(self, delta_std):
        self.delta_std = delta_std

    def onedirction_rollout_multi_single_Cases(self, w_policy, ob_mean, ob_std, select_fault_cases_tuples, evaluate=False):
        rollout_rewards, deltas_idx = [], []
        steps = 0											

        # use RS to collect observation states from SingleRolloutSlavers; restart each iter
        self.RS = RunningStat(shape=(self.OB_DIM,))

        if evaluate:
            deltas_idx.append(-1)
            weights_id = w_policy

            reward_ids = []	 
														
            for i in range(len(select_fault_cases_tuples)):
                fault_tuple_tmp=select_fault_cases_tuples[i]
                #fault_tuple_idx=PF_FAULT_CASES_ALL.index(fault_tuple_tmp)
                reward_ids.append(self.single_rollout_slavers[i].single_rollouts(fault_tuple=fault_tuple_tmp,
                                    weights=weights_id, ob_mean=ob_mean, ob_std=ob_std)) 

            results = reward_ids
            reward_list = []
            step_list = []
            for result in results:
                reward_list.append(result['reward'])
                step_list.append(result['step'])
            reward_ave = np.mean(reward_list)
            steps_max = max(step_list)			

			# ----- delete the weight in the global table in ray before return
            del weights_id	
			
            return {'reward_ave': reward_ave, 'reward_list': reward_list}

        else:
            idx, delta = self.deltas.get_delta(w_policy.size)
            delta = (self.delta_std * delta).reshape(w_policy.shape)
            deltas_idx.append(idx)

            pos_rewards_list = []
            pos_steps_list = []
            neg_rewards_list = []
            neg_steps_list = []
            time_list = []

            t1 = time.time()
            # compute reward and number of timesteps used for positive perturbation rollout    
            w_pos_id = w_policy + delta
            pos_list_ids = [self.single_rollout_slavers[i].single_rollout(
                fault_tuple=select_fault_cases_tuples[i], weights=w_pos_id, ob_mean=ob_mean, ob_std=ob_std) \
                for i in range(self.num_slaves)]
            
            pos_results  = pos_list_ids
            for result in pos_results:
                pos_rewards_list.append(result['reward'])
                pos_steps_list.append(result['step'])

                # if b_debug_timing:
                #     time_list.append(result['time'])
					
            pos_reward = np.mean(pos_rewards_list)
            pos_steps = max(pos_steps_list)
            # get RS of SingleRolloutSlavers
            pos_RS_ids = [self.single_rollout_slavers[i].return_RS() for i in range(self.num_slaves)]
            pos_RSs = pos_RS_ids

            # compute reward and number of timesteps used for negative pertubation rollout             
            w_neg_id = w_policy - delta
            neg_list_ids = [self.single_rollout_slavers[i].single_rollout(
                fault_tuple=select_fault_cases_tuples[i], weights=w_neg_id, ob_mean=ob_mean, ob_std=ob_std) \
                for i in range(self.num_slaves)]
            neg_results  = neg_list_ids
            for result in neg_results:
                neg_rewards_list.append(result['reward'])
                neg_steps_list.append(result['step'])
                # if b_debug_timing:
                #     time_list.append(result['time'])
					
            neg_reward = np.mean(neg_rewards_list)
            neg_steps = max(neg_steps_list)												
            neg_RS_ids = [self.single_rollout_slavers[i].return_RS() for i in range(self.num_slaves)]
            neg_RSs = neg_RS_ids
            t2 = time.time()
			
            # update RS from SingleRolloutSlavers
            for pos_RS in pos_RSs:
                self.RS.update(pos_RS)
            for neg_RS in neg_RSs:
                self.RS.update(neg_RS)

            steps += pos_steps + neg_steps
            rollout_rewards.append([pos_reward, neg_reward])
			# ----- delete the positive and negative weights in the global table in ray before return
            del w_pos_id
            del w_neg_id
			
            t3 = time.time()
			
            # if b_debug_timing:			
            #     return {'deltas_idx': deltas_idx, 'rollout_rewards': rollout_rewards, "steps": steps, 'time': t2-t1, 't1': t1, 't2': t2, 't3': t3,\
            #         'slavetime': time_list, 'slavestep': pos_steps_list+neg_steps_list}
            # else:			
            return {'deltas_idx': deltas_idx, 'rollout_rewards': rollout_rewards, "steps": steps, 'time': t2-t1}
                    #'slavetime': time_list, 'slavestep': pos_steps_list+neg_steps_list}

    def get_filter(self):
        return self.RS
        

    def close_env(self):
        close_ids = [slave.close_env() for slave in self.single_rollout_slavers]
        return




class PARS(erl.ExaAgent):
    """Parallel Agumented Random Search agent.
    Inherits from ExaAgent base class.
    """

    def __init__(self, env, is_learner):
        
        self.is_learner = is_learner
        self.agent_comm = ExaComm.agent_comm
        self.learner_comm = ExaComm.learner_comm

        # Get the environnemt
        self.env = env

        # get the dimension of the observations and action buses
        # in case of actual environment uncomment below

        # self.ob_dim = env.observation_space.shape[0]
        # self.ac_dim = env.action_space.shape[0]
        
        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.shape[0]

        print(self.ob_dim,self.ac_dim,"****")
        #grab a fault tuple from FAULT_CASES
        self.SELECT_PF_PER_DIRECTION = 1
        self.POWERFLOW_CANDIDATES = list(range(1)) 
        self.SELECT_FAULT_BUS_PER_DIRECTION = 5
        self.FAULT_BUS_CANDIDATES = list(range(5))
        self.FAULT_START_TIME = 1.0
        self.FTD_CANDIDATES = [0.08]
        self.PF_FAULT_CASES_ALL = [(self.POWERFLOW_CANDIDATES[k], self.FAULT_BUS_CANDIDATES[i], self.FAULT_START_TIME, self.FTD_CANDIDATES[j]) 
                for k in range(len(self.POWERFLOW_CANDIDATES))
                for i in range(len(self.FAULT_BUS_CANDIDATES)) 
				for j in range(len(self.FTD_CANDIDATES))]



        
        
        self.params = self.CreateParams()

        # Define the Policy parameters
        self.policy_params = {'type': self.params['policy_type'],
                            'policy_network_size': self.params['policy_network_size'],
                            'ob_dim': self.ob_dim,
                            'ac_dim': self.ac_dim}
         # initialize policy
        print('Initializing policy.', flush=True)
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
        print('Initializing optimizer.')
        self.optimizer = SGD(self.w_policy, self.params["step_size"])
        print("Initialization of ARS complete.")

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
        
        # Each worker/actor picks a delta from the random stream.
        self.deltas_idx = []
        self.delta = None
        self.worker_seed_id = np.random.randint(0,high=10000)
        self.set_delta(self.worker_seed_id)

        # Collect all +ve/-ve reward means
        self.pos_neg_meanreward = []
        
        
        # The async_learner.py provide the parallelization 
        # for 1- H iterations...in Alg.-1 of paper
        # ACCELERATED DERIVATIVE-FREE DEEP REINFORCEMENT LEARNING

        # Total Number of cases a actor needs to perform
        # assumption: Num_actors ==  Num_delta (Number of deltas)
        # self.Num_cases =  Num_actors x Num_perturb_direc x N_fault 
        self.Num_actors = 1 # TODO: self.agent_comm.size - self.learner_comm.size
        self.Num_pertub = 2  #  This is +ve and -ve perturbations direction
        self.Num_faults = len(self.PF_FAULT_CASES_ALL)     

        self.epsilon = ExaGlobals.lookup_params('epsilon')   
    

        # We define the internal step counter
        # This will be always rest to zero by the set-weight call
        #
        self.internal_step_count = 0

        # Initializat positive and negative reward
        self.pos_rew, self.neg_rew = [], []

        # This is to collect all the batches of the worker 
        # by the learner.
        self.all_actorbatch = []

        # This count as one-episode considering the +ve/-ve pertub and Number of faults cases.
        # self.N_cases_beforeUpdate = self.Num_pertub*self.Num_faults*self.params['rollout_length']
        self.Num_CasesbeforeUpdate = self.Num_pertub*self.Num_faults

        # This is the number of steps which happens before the update.
        # This is example: 2perturb x 5 cases x 80 n_steps = 800  
        self.Num_StepBeforeUpdate = self.Num_CasesbeforeUpdate * self.params['rollout_length'] 


        # Flag to use by set weight update and 
        self.set_flag = 1

    def CreateParams(self):
        param = {}       
        size = ExaGlobals.lookup_params('policy_network_size')
        param['policy_network_size'] = list([size,size])
        param['delta_std'] = ExaGlobals.lookup_params('delta_std')
        param['decay'] = ExaGlobals.lookup_params('decay')
        param['rollout_length'] =  ExaGlobals.lookup_params('n_steps')
        param['seed'] = ExaGlobals.lookup_params('seed')
        param['policy_type'] = ExaGlobals.lookup_params('model_type')
        param['step_size'] = ExaGlobals.lookup_params('step_size')
        casestorunperdirct = self.SELECT_PF_PER_DIRECTION * len(self.FTD_CANDIDATES) * self.SELECT_FAULT_BUS_PER_DIRECTION
        param['onedirection_numofcasestorun'] = casestorunperdirct
        
        return param


    def set_delta(self,seed):
        deltas_id = create_shared_noise()
        self.deltas = SharedNoiseTable(deltas_id,self.params["seed"] + 7 * seed)
        idx, delta = self.deltas.get_delta(self.w_policy.size)
        self.delta = (self.delta_std * delta).reshape(self.w_policy.shape)
        self.deltas_idx.append(idx)


    def get_weights(self):
        print("PARS:  getting weights episode:", self.env.workflow_episode)
        self.w_policy = self.policy.get_weights()
        return self.w_policy
    
    def calc_mean_pos_neg_reward(self):
        assert len(self.pos_rew) != 0 , "Positive Peturbation reward list empty"
        assert len(self.neg_rew) != 0 , "Negative Peturbation reward list empty"
        
        mean_pos_reward = np.mean(self.pos_rew)
        mean_neg_reward = np.mean(self.neg_rew)

        self.pos_neg_meanreward.append([mean_pos_reward,mean_neg_reward])
        return 
    
    def set_weights(self, weights):
      
        
        if (self.env.workflow_episode+1) % self.Num_CasesbeforeUpdate == 0 and (self.internal_step_count+1) >= self.Num_StepBeforeUpdate:
            
            # This flag is used to ensure that once the above if condition is met the weights are updated only 
            # for the first time on the increase of the episode counter and else use the same weights throughout
            # the episode.
            print("set flag status = ",self.set_flag)
            if self.set_flag == 1:
                # This assertion will ensure that the update of weights are called only after finishing 
                # the required number of steps ...
                print("Inside set weights...",self.internal_step_count,self.env.workflow_episode)
                assert (self.internal_step_count+1) == self.Num_CasesbeforeUpdate * self.params['rollout_length']
    
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
                
                # reset the internal_step_count
                self.internal_step_count  = 0
                # reset the flag
                self.set_flag = 0
                print("From inside set_weight: setting set flag status == ",self.set_flag)
            else:
                pass
        else:
           
            # Even count mean run with positive perturb
            if self.env.workflow_episode % 2 == 0:
                # update with the positive 
                w_pos_id = self.w_policy + self.delta
                self.policy.update_weights(w_pos_id)
                print("Setting positive weights.. Episode:: ",self.env.workflow_episode)
            # Odd count mean run with negative perburb 
            else:
                # update with the negative perturb 
                w_pos_id = self.w_policy - self.delta
                self.policy.update_weights(w_pos_id)
                print("Setting negative weights..!  Episode:: ",self.env.workflow_episode)
                   
    def action(self,state):
        ob = np.asarray(state, dtype=np.float64)

        # Calculate the normalized observation.
        normal_ob = (ob - self.ob_mean) / (self.ob_std + 1e-8)

        # This counter is increased here since
        # the action is followed by the step of the environment  
        # an actor fucntion of sync_learner.py.
        self.internal_step_count += 1

        return self.policy.act(normal_ob), self.policy_params['type']


    def train_step(self,g_hat):
        print("Euclidean norm of update step:", np.linalg.norm(g_hat))
        self.w_policy -= self.optimizer._compute_step(g_hat, self.step_size).reshape(self.w_policy.shape)
        print('g_hat shape, w_policy shape:',np.asarray(g_hat).shape,self.w_policy.shape)

        # update the policy with new weights...
        self.policy.update_weights(self.w_policy)
        return

    def train(self,batch):

        # print("Check if batch is None::",batch == None)
        # print("Condition Train ::" , (self.env.workflow_episode+1) % self.Num_CasesbeforeUpdate == 0, (self.internal_step_count+1) >= self.Num_StepBeforeUpdate)
        
        if (self.env.workflow_episode+1) % self.Num_CasesbeforeUpdate == 0 and (self.internal_step_count+1) >= self.Num_StepBeforeUpdate:
    
            # check if all actor batches are appended  
            if len(self.all_actorbatch) != self.Num_actors:
                
                print("appending the batch and returning NONE")
                self.all_actorbatch.append(batch)

                return None

            else:
                # use the self.all_actorbatch once all the actors have update 
                # the actorbatch list.
                assert len(self.all_actorbatch) > 0 , "one or more actors have not finished N_cases_beforeUpdate"
                
                rollout_rewards, deltas_idx, deltas_actor = [], [],[]
                for i, actorbatch in enumerate(self.all_actorbatch):
                    rollout_rewards += actorbatch['rollout_rewards']
                    deltas_idx += actorbatch['deltas_idx']
                    deltas_actor.append(actorbatch['deltas'])
                    
                    
                # This is the collection of all the rewards...
                rollout_rewards = np.asarray(rollout_rewards)

                max_rewards = np.max(rollout_rewards, axis=1)
                
                # if self.deltas_used > self.num_deltas:
                #     self.deltas_used = self.num_deltas
                
                #  select top performing deltas;  95 percentile  data...
                idx = np.arange(max_rewards.size)[max_rewards >= np.percentile(max_rewards, 0.95)]

                deltas_idx = np.array(deltas_idx)[idx]
                rollout_rewards = np.array(rollout_rewards)[idx, :]
                deltas_actor = np.array(deltas_actor)[idx[0]]
               
                # normalize rewards by their standard deviation
                if np.std(rollout_rewards) > 1:
                    rollout_rewards /= np.std(rollout_rewards)

                # aggregate rollouts to form g_hat, the gradient used to compute SGD step
                g_hat, count = batched_weighted_sum(rollout_rewards[:, 0] - rollout_rewards[:, 1],
                                                        (deltas_actor.get(idx, self.w_policy.size)
                                                        for idx in deltas_idx),
                                                        batch_size=500)
                g_hat /= deltas_idx.size

                self.step_size *= self.decay # This updated value is used at train_step 
                self.delta_std *= self.decay # This is used to update delta_std in one-worker instance.
                # This batch comes from the generate function...
                # g_hat = batch[0]
                self.train_step(g_hat)
                
                # Loss metric.
                l2_norm = np.linalg.norm(g_hat)
                return l2_norm

        else:
            # Don't train since the number of perturb and number of all faults
            # cases have not finished.
            return None 

        
    


    def generate_data(self):
        # 
        batch = {}

        # print( "In Generate Data:",(self.env.workflow_episode+1) % self.Num_CasesbeforeUpdate == 0 , (self.internal_step_count+1) >= self.Num_StepBeforeUpdate)
        # print("From Generate Data", self.internal_step_count+1,self.Num_StepBeforeUpdate)
        # This should return 
        if  (self.env.workflow_episode+1) % self.Num_CasesbeforeUpdate == 0 and (self.internal_step_count+1) >= self.Num_StepBeforeUpdate:
            
            # Call the pos_neg_meanreward calc
            self.calc_mean_pos_neg_reward()

            batch['rollout_rewards'] = self.pos_neg_meanreward
            batch['deltas_idx'] = self.deltas_idx
            batch['deltas'] = self.deltas

            # Reset the positive and negative reward list for 
            self.pos_rew, self.neg_rew = [], []
            print("*PARS step Count:", self.internal_step_count)
            yield batch
        else:
            self.set_flag = 1 # reset the set_weight flag
            print("*PARS step Count:", self.internal_step_count)
        
            yield None

    
    def remember(self, state, action, reward, next_state, done):
        # self.memory.store(state, action, reward, next_state, done)
        # JS: What does onedirction_rollout_multi_single_Cases care about
        # you care about RS and reward...
        # Store it here!
        # Probable need self.step

        self.RS.push(state)
        # positive perturb policy returning
        if self.env.workflow_episode % self.Num_pertub == 0:
            self.pos_rew.append(reward)
        # negative perturb policy returning
        else:
            self.neg_rew.append(reward)

        
    
    def target_train(self):
        return self.policy.get_weights

    def update(self):
        print("Implement update method in ARS.py")
        return 

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
        