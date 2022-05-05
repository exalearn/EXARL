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
import time
import os
import math
import json
import csv
import random
from turtle import update
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
from exarl.utils import log
from exarl.utils.introspect import introspectTrace
from tensorflow.compat.v1.keras.backend import set_session


from exarl.envs.env_vault.Hadrec_dir.src.policy_LSTM import *
from exarl.envs.env_vault.Hadrec_dir.src.utils import *
from exarl.envs.env_vault.Hadrec_dir.src.logz import *

from exarl.envs.env_vault.Hadrec_dir.src.optimizers import *

from exarl.agents.agent_vault._replay_buffer import ReplayBuffer
from exarl.utils.globals import ExaGlobals

logger = ExaGlobals.setup_logger(__name__)

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

        # Get the environnemt
        self.env = env

        # get the dimension of the observations and action buses
        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.shape[0]
        
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

    
        self.buffer_capacity = ExaGlobals.lookup_params('buffer_capacity')
        self.batch_size = ExaGlobals.lookup_params('batch_size')

        # JS: You don't really want this.  What you want is RS + onedirction_rollout_multi_single_Cases
        self.memory = ReplayBuffer(self.buffer_capacity, self.ob_dim, self.ac_dim)
        
        
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
            self.policy = FullyConnectedNeuralNetworkPolicy(self.policy_params, seed)
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

        # Initialize the Running stats...
        self.RS = RunningStat(shape=(self.ob_dim,))

        self.deltas_used = self.params['delta_used']
        self.num_deltas = self.params['n_directions']
        self.step_size= self.params['step_size']
        self.delta_std = self.params['delta_std']
        self.decay = self.params['decay']

        deltas_id = create_shared_noise()
        self.deltas = SharedNoiseTable(deltas_id)
        
        self.worker_seed_id = np.random.randint(0,high=100)
        self.one_direction_workers = OneDirectionWorker(self.env,self.params["seed"] + 7 * self.worker_seed_id,
                                      policy_params=self.policy_params,
                                      deltas=deltas_id,
                                      rollout_length=self.params["rollout_length"],
                                      delta_std=self.delta_std,
                                      PF_FAULT_CASES_ALL=self.PF_FAULT_CASES_ALL,
                                      OB_DIM=self.ob_dim																						  
                                      ) 



    def CreateParams(self):
        param = []        
        param['n_iter'] = ExaGlobals.lookup_params('n_iter')
        param['n_directions'] = ExaGlobals.lookup_params('n_directions')
        param['deltas_used'] = ExaGlobals.lookup_params('deltas_used')
        param['policy_network_size'] = ExaGlobals.lookup_params('policy_network_size')
        param['save_per_iter'] = ExaGlobals.lookup_params('step_size')
        param['delta_std'] = ExaGlobals.lookup_params('delta_std')
        param['decay'] = ExaGlobals.lookup_params('decay')
        param['rollout_length'] = ExaGlobals.lookup_params('rollout_length')
        param['seed'] = ExaGlobals.lookup_params('seed')
        param['policy_type'] = ExaGlobals.lookup_params('policy_type')
        param['tol_p'] = ExaGlobals.lookup_params('tol_p')
        param['tol_steps'] = ExaGlobals.lookup_params('tol_step')
        param['dir_path'] = ExaGlobals.lookup_params('dir_path')
        param['step_size'] = ExaGlobals.lookup_params('step_size')
        param['cores'] = ExaGlobals.lookup_params('cores')

        casestorunperdirct = self.SELECT_PF_PER_DIRECTION * len(self.FTD_CANDIDATES) * self.SELECT_FAULT_BUS_PER_DIRECTION
       
        param['onedirection_numofcasestorun'] = casestorunperdirct
        
        return param

    def aggregate_tiles():
        """
        This function should be called at first fault senario of a rollout
        """
        # JS: Don't forget to reset self.step...
        if self.RS_counter % self.PF_FAULT_CASES_ALL == 0:
            print("Reseting Running Stat")
            self.RS = RunningStat(shape=(self.ob_dim,))
            self.ob_mean = self.RS.mean
            self.ob_std = self.RS.std
        else:
            self.RS.update((self.one_direction_workers.get_filter()))

    def get_weights(self):
        print("PARS getting weights")
        return self.w_policy
        
    def set_weights(self, weights):
        print("PARS setting weights")
        self.policy.update_weights(weights)
        # JS: WE NEED TO DOUBLE CHECK THIS IS CORRECT.
        self.aggregate_tiles()
            
    def action(self,state):
        ob = np.asarray(state, dtype=np.float64)
        self.RS.push(ob)
        normal_ob = (ob - self.ob_mean) / (self.ob_std + 1e-8)
        return self.policy.act(normal_ob)


    def train_step(self,g_hat):
        print("Euclidean norm of update step:", np.linalg.norm(g_hat))
        self.w_policy -= self.optimizer._compute_step(g_hat, self.step_size).reshape(self.w_policy.shape)
        print('g_hat shape, w_policy shape:',np.asarray(g_hat).shape,self.w_policy.shape)
        return

    def train(self,batch):
        # JS: batch should be g_hat passed from generage_data
        if batch[0] % self.PF_FAULT_CASES_ALL :
            rewardlist = []
            last_reward = 0.0 

            self.step_size *= self.decay # This updated value is used at train_step 
            self.delta_std *= self.decay # This is used to update delta_std in one-worker instance.

            # Each actor need to update seed and delta_std
            self.one_direction_workers.update_delta_std(delta_std=self.delta_std)


            # This batch comes from the generate function...
            g_hat = batch[0]
            self.train_step(g_hat)

        # If certain 
        return
    
    

    def EvaluateCall(self):

        # This evaluation implementations assume that the
        # The 

        self.one_direction_workers.onedirction_rollout_multi_single_slavers(policy_id,ob_mean,ob_std,select_fault_cases_tuples=self.PF_FAULT_CASES_ALL)
    


        return reward_eval_avr, rewards_list

    def generate_data(self):
        # JS: the second half (evaluate=false) of onedirction_rollout_multi_single_Cases should be done here
        # return G_hat

        # i : here is based on actor rank 
        # for now we chose it from a random generator
        self.worker_seed_id = np.random.randint(0,high=100)
        
        select_faultbus_num = self.SELECT_FAULT_BUS_PER_DIRECTION
        select_faultbuses_id = np.random.choice(self.FAULT_BUS_CANDIDATES, size=select_faultbus_num, replace=False)
        print ("select_faultbuses_id:  ", select_faultbuses_id)
        select_pf_num = self.SELECT_PF_PER_DIRECTION
        select_pfcases_id = np.random.choice(self.POWERFLOW_CANDIDATES, size=select_pf_num, replace=False)
        print ("select_pfcases_id:  ", select_pfcases_id)
        select_fault_cases_tuples = [(select_pfcases_id[k], select_faultbuses_id[i], self.FAULT_START_TIME, self.FTD_CANDIDATES[j]) \
                                        for k in range(len(select_pfcases_id))
                                        for i in range(len(select_faultbuses_id)) 
                                        for j in range(len(self.FTD_CANDIDATES))]
										
        select_fault_cases_tuples_id = select_fault_cases_tuples
        print ("select_fault_cases_tuples:  ", select_fault_cases_tuples) 

        policy_id = self.w_policy
        ob_mean = self.RS.mean
        ob_std = self.RS.std

        rollout_id_list = self.one_direction_workers.onedirction_rollout_multi_single_Cases(policy_id, ob_mean, ob_std, select_fault_cases_tuples_id,evaluate=False)


        # gather results
        results_list = rollout_id_list
        rollout_rewards, deltas_idx = [], []
		
        
        iworktmp = 0
        for result in results_list:
            self.timesteps += result["steps"]										  
            deltas_idx += result['deltas_idx']
            rollout_rewards += result['rollout_rewards']
            iworktmp += 1				   

        print("rollout_rewards shape:", np.asarray(rollout_rewards).shape)
        print("deltas_idx shape:", np.asarray(deltas_idx).shape)

        deltas_idx = np.array(deltas_idx)
        rollout_rewards = np.array(rollout_rewards, dtype=np.float64)
        
        print('Maximum reward of collected rollouts:', rollout_rewards.max())
    
        # update RS from all workers
        # for j in range(self.num_workers):
        self.RS.update((self.one_direction_workers.get_filter()))

        # select top performing directions if deltas_used < num_deltas
        max_rewards = np.max(rollout_rewards, axis=1)
        
        if self.deltas_used > self.num_deltas:
            self.deltas_used = self.num_deltas

        idx = np.arange(max_rewards.size)[max_rewards >= np.percentile(max_rewards, 100 * ( 
                    1 - (self.deltas_used / self.num_deltas)))]
        deltas_idx = deltas_idx[idx]
        rollout_rewards = rollout_rewards[idx, :]

        # normalize rewards by their standard deviation
        if np.std(rollout_rewards) > 1:
            rollout_rewards /= np.std(rollout_rewards)

        # aggregate rollouts to form g_hat, the gradient used to compute SGD step
        g_hat, count = batched_weighted_sum(rollout_rewards[:, 0] - rollout_rewards[:, 1],
                                                  (self.deltas.get(idx, self.w_policy.size)
                                                   for idx in deltas_idx),
                                                  batch_size=500)
        g_hat /= deltas_idx.size
        # del the policy weights, ob_mean and ob_std in the object store
        del policy_id
        del ob_mean
        del ob_std
        del select_fault_cases_tuples_id																	
                
        batch = [g_hat]
        return batch
    
    def remember(self, state, action, reward, next_state, done):
        # self.memory.store(state, action, reward, next_state, done)
        # JS: What does onedirction_rollout_multi_single_Cases care about
        # you care about RS and reward...
        # Store it here!
        # Probable need self.step
        self.step += 1

    def update(self):
        print("Implement update method in ARS.py")
        return 

    def load(self):
        print("Implement load method in ARS.py")
        return 

    def save(self):
        print("Implement save method in ARS.py")
        return 


    def monitor(self):
        print("Implement monitor method in ARS.py")
    

    def has_data(self):
        """Indicates if the buffer has data of size batch_size or more

        Returns:
            bool: True if replay_buffer length >= self.batch_size
        """
        print("Implement has_data method in ARS.py")
        