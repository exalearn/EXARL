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


#
# The ARS agent implementation is motivated and adopted from 
# the stable-baseline3 Contrib and the Seagul github repository (https://github.com/sgillen/seagul)
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import time
import os
import math
import json
import csv
import random
import tensorflow as tf
import sys
import gym
import exarl as erl
from exarl.base.comm_base import ExaComm
from collections import deque
from datetime import datetime
import numpy as np
from numpy.random import default_rng


import torch
import torch.nn as nn



from exarl.utils.globals import ExaGlobals
logger = ExaGlobals.setup_logger(__name__)
import h5py


class MLPModel(nn.Module):

    def __init__(self, in_size, out_size, num_l , l_size, 
                 activ=nn.ReLU,
                 out_activ=nn.Identity,
                 squash_out = True):
        super(MLPModel,self).__init__()

        self.activation = activ()
        self.out_activation = out_activ()
        self.squash_out = squash_out
        self.state_means = torch.zeros(in_size, requires_grad=False)
        self.state_std = torch.ones(in_size, requires_grad=False)
        
        if num_l == 0:
            self.mlp_layers = []
            self.mlp_out_layer = nn.Linear(in_size, out_size,bias=True)
        else:
            self.mlp_layers = nn.ModuleList([nn.Linear(in_size, l_size, bias=True)])
            self.mlp_layers.extend([nn.Linear(l_size, l_size,bias=True) for _ in range(num_l-1)])
            self.mlp_out_layer = nn.Linear(l_size, out_size,bias=True)

    
    def forward(self,data):
        # Normalize the input data with mean and std
        data = (torch.as_tensor(data, dtype=self.state_means.dtype) - self.state_means) / self.state_std
       
        for layer in self.mlp_layers:
            data = self.activation(layer(data.float()))

        # Last Layer of the network output..
        out = self.out_activation(self.mlp_out_layer(data))

        # Pass the output via tanh activation function
        if self.squash_out:
            out = nn.Tanh()(out)
        return out


class PARS_V1(erl.ExaAgent):
    def rankPrint(self, *args):
        print("AGENT_RANK:", self.agent_comm.rank, "IS_LEARNER:", self.is_learner, *args, flush=True)  
    
    """Parallel Agumented Random Search agent.
    Inherits from ExaAgent base class.
    """

    def __init__(self, env, is_learner):
        
        self.is_learner = is_learner
        self.agent_comm = ExaComm.agent_comm
        self.learner_comm = ExaComm.learner_comm

        # Get the environnemt
        self.env = env

        # self.ob_dim = env.observation_space.shape[0]
        # self.ac_dim = env.action_space.shape[0]
        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.shape[0]

        self.params = self.CreateParams()

        # These parameter are used for delta setting.
        self.step_size= self.params['step_size']
        self.n_delta = self.params['n_delta']
        self.delta_std = self.params['delta_std']
        self.n_top = self.params['n_top']

        # This dummy, it is here just for the sake of workflow purpose.
        self.epsilon = ExaGlobals.lookup_params('epsilon')   
        self.n_steps = ExaGlobals.lookup_params('n_steps')

        # Create the Model
        self.model = self.CreateModel()

       
        # Fetch the Weights from the model... 
        W_torch = torch.nn.utils.parameters_to_vector(self.model.parameters())
        self.W_flat = W_torch.detach().numpy()
        
        # Initial Weights of the Model...
        self.W_flat_init =  W_torch.detach().numpy()

        # Initialize the states mean and standard deviation.        
        self.state_mean = np.zeros(self.ob_dim)
        self.state_std = np.ones(self.ob_dim)

        # Range function call from numpy random.
        self.rng = default_rng()  
        self.deltas = None # This will be set when generate_PM_W is invoked.self

        # Storing list per step...reset will happen at the end of an episode.
        # in the generate data function...
        self.state_list = []
        self.act_list = []
        self.reward_list = []

        self.raw_rew_hist = []
        self.rolling_ep_rew_mean = []

        # Total_steps 
        self.total_steps = 0
        self.total_episode =0

    def CreateParams(self):
        param = {}       
        size = ExaGlobals.lookup_params('policy_network_size')
        param['policy_network_size'] = list([size,size])
        param['delta_std'] = ExaGlobals.lookup_params('delta_std')
        param['n_delta'] = ExaGlobals.lookup_params('n_delta')
        param['n_top'] = ExaGlobals.lookup_params('n_top')
        param['decay'] = ExaGlobals.lookup_params('decay')
        param['rollout_length'] =  ExaGlobals.lookup_params('n_steps')
        param['seed'] = ExaGlobals.lookup_params('seed')
        param['policy_type'] = ExaGlobals.lookup_params('model_type')
        param['step_size'] = ExaGlobals.lookup_params('sgd_step_size')
        param['num_l'] =  ExaGlobals.lookup_params('num_layers')
      
        param['l_size'] = size

        return param
        

    def CreateModel(self):
        # initialize policy
        if self.params['policy_type'] == 'linear':
            model = MLPModel(self.ob_dim, self.ac_dim, 
                            self.params['num_l'],
                            self.params['l_size'], 
                            activ=nn.Identity,
                            out_activ=nn.Identity)
        elif self.params['policy_type'] == 'nonlinear':
            model = MLPModel(self.ob_dim, self.ac_dim, 
                            self.params['num_l'],
                            self.params['l_size'], 
                            activ=nn.ReLU,
                            out_activ=nn.Identity)
        elif self.params['policy_type'] == 'LSTM':
            raise NotImplementedError										 																		
        else:
            raise NotImplementedError

        return model

    def Generate_pmW(self):
        # Note:
        # The self.W_flat_init is coming from the agent instantiations.
        # The self.W_flat_init will be updated by the train on the call of learner
        # after the actor finishes the loop over all +/- weight.
        # self.rankPrint(f" Generate_pmW \n {self.W_flat_init} ")
        n_param = self.W_flat_init.shape[0]
        self.deltas = self.rng.standard_normal((self.n_delta, n_param))
        pm_W = np.concatenate((self.W_flat_init+(self.deltas*self.delta_std), self.W_flat_init-(self.deltas*self.delta_std)))
        return pm_W

    def Mean_update(self,data,cur_mean, cur_step):
        updated_step = data.shape[0]
        updated_mean = (np.mean(data,0) * updated_step + cur_mean* cur_step)/ (cur_step + updated_step) 
        return updated_mean
    
    def Std_update(self,data, cur_std, cur_steps):
        updated_step = data.shape[0]
        batch_var = np.var(data, 0)

        if np.isnan(batch_var).any():
            return cur_std
        else:
            cur_var = cur_std ** 2
            new_var = np.std(data, 0) ** 2
            new_var[new_var < 1e-4] = cur_var[new_var < 1e-4]
            updated_std = np.sqrt((new_var * updated_step + cur_var * cur_steps) / (cur_steps + updated_step)) 
            return updated_std
        
    def get_weights(self):
        return self.W_flat_init
    
    def set_weights(self, weights):
        
        # self.rankPrint(f"Set Weights... {type(weights)}")
        # Set the model weights ...
        torch.nn.utils.vector_to_parameters(torch.tensor(weights,requires_grad=False), self.model.parameters())
        
        # Collect the set weights in the W_flat variable...
        # This will ensure that the get_weights call returns correct weights..
        W_torch = torch.nn.utils.parameters_to_vector(self.model.parameters())
        self.W_flat = W_torch.detach().numpy()
        
        
        self.model.state_means = torch.from_numpy(self.state_mean)
        self.model.state_std = torch.from_numpy(self.state_std)

        self.model.float()
        # self.rankPrint(f" Weights-set from set weight:, {self.W_flat}")
        return
                           
    def action(self,state):
        obs = np.asarray(state, dtype=np.float32)
        # self.rankPrint(f"{obs}, {type(obs)} {obs.reshape(1,-1)}......")
        obs = torch.from_numpy(obs.reshape(1,-1)).float()
        act = self.model(obs).detach().numpy()
        # self.rankPrint(f"{act}, {type(act)}, {type(act.detach().numpy())}")
        return act , self.params['policy_type']

    def train(self,batch):
        
        # self.rankPrint("In Train Step....")
        # self.rankPrint("")
        # states = np.array([]).reshape(0,self.ob_dim)
        # p_returns = []
        # m_returns = []
        # top_returns = []

        # for p_result, m_result in zip(batch[:self.n_delta], batch[self.n_delta:]):
        #     ps, pr = p_result['state_arr'], p_result['reward_sum']
        #     ms, mr = m_result['state_arr'], m_result['reward_sum']

        #     states = np.concatenate((states, ms, ps), axis=0)
        #     p_returns.append(pr)
        #     m_returns.append(mr)
        #     top_returns.append(max(pr,mr))
        rollout_ep_rew_mean = batch['rollout/ep_rew_mean']
        top_returns = batch['top_returns']
        p_returns = batch['p_returns']
        m_returns = batch['m_returns']
        states = batch['states']

        # self.rankPrint(f"Epi-count - {self.total_episode} top_returns; \n {top_returns}")
        
        top_idx = sorted(range(len(top_returns)), key=lambda k: top_returns[k], reverse=True)[:self.n_top]
        p_returns = np.stack(p_returns)[top_idx]
        m_returns = np.stack(m_returns)[top_idx]
        # l_returns = np.stack(l_returns)[top_idx]

        if self.total_episode % 10 == 0:
            self.rankPrint(f"Episode {self.total_episode} rollout/ep_rew_mean :: {rollout_ep_rew_mean}")
            # self.rankPrint(f"Top Returns Mean : {np.stack(top_returns)[top_idx].mean()} ")
            # self.rankPrint(f"+/- reward Mean : {(p_returns.mean() + m_returns.mean())/2}")
            # print(f"{self.total_episode} : mean return: {top_returns.mean()}, top_return: {top_returns.max()}")
        
        
        self.raw_rew_hist.append(np.stack(top_returns)[top_idx].mean())
        self.rolling_ep_rew_mean.append(rollout_ep_rew_mean)


        return_std = np.concatenate((p_returns, m_returns)).std() + 1e-6
        step_size = self.step_size / (self.n_top * return_std + 1e-6)

        p_2 = np.sum((p_returns - m_returns)*self.deltas[top_idx].T, axis=1)
        
        
        # self.rankPrint(f" Train....W_flat_init before update \n {self.W_flat_init} ")

        # Update the weights based on the ARS differencing scheme. 
        self.W_flat_init = self.W_flat_init +  step_size * p_2

        # self.rankPrint(f" Train....W_flat_init after update \n {self.W_flat_init} ")

        ep_steps = states.shape[0]
        self.state_mean = self.Mean_update(states, self.state_mean, self.total_steps)
        self.state_std = self.Std_update(states, self.state_std, self.total_steps)

        self.total_steps += ep_steps
        self.total_episode += 1
         
        return None 

    def remember(self, state, action, reward, next_state, done):
        # self.rankPrint(f">>>><<< {len(reward)}, {reward}")
        # self.rankPrint(f" >>> {len(state)},  {type(state)}, {len(action)}, {type(action)}, {len(reward)}, {type(reward)}")
        
        # self.rankPrint(f"{len(state)}, {state}, {type(state)}")
        state = state.reshape(self.ob_dim,-1)

        self.state_list.append(state)
        self.act_list.append(action)
        self.reward_list.append(reward)
        return

    def generate_data(self):
        batch_data = {}

        self.reward_list = np.array(self.reward_list).reshape(self.n_delta*2, self.n_steps)

        # sum over all the n_steps in a single episode run
        rewardSum_perEpisode = np.sum(self.reward_list,axis=1)

        top_returns = []
        for pr, mr in zip(rewardSum_perEpisode[:self.n_delta], rewardSum_perEpisode[self.n_delta:]):
            top_returns.append(max(pr,mr))

        # self.rankPrint(f"Top Rewards..from +/- comparison :: \n {(top_returns)}")
        # self.rankPrint(f"{np.stack(self.state_list).shape}")
        # self.rankPrint(f"{np.stack(self.state_list)}")

        # states = np.concatenate((states, ms, ps), axis=0)
        # p_returns.append(pr)
        # m_returns.append(mr)
        # top_returns.append(max(pr,mr))
        # self.rankPrint(f'Shape...: {(self.reward_list.flatten()).shape}')
        # self.rankPrint(f'Shape...: {np.mean(self.reward_list)}')
        # self.rankPrint(f'Shape...: { np.mean(rewardSum_perEpisode)}')

        batch_data['rollout/ep_rew_mean'] = np.mean(rewardSum_perEpisode)
        batch_data['top_returns'] = top_returns
        batch_data['p_returns'] = rewardSum_perEpisode[:self.n_delta]
        batch_data['m_returns'] = rewardSum_perEpisode[self.n_delta:]
        batch_data['states'] = np.stack(self.state_list).squeeze()
        batch_data['act_arr'] = np.stack(self.act_list)
        

        
        # Reset the storing list
        self.state_list = []
        self.act_list = []
        self.reward_list = []
        
        # Update the mean and sta
        yield batch_data
    
    def target_train(self):
        return self.model.parameters()

    def update_target(self):
        pass
        # print("Implement update method in ARS.py")
        return 

    def load(self):
        print("Implement load method in ARS.py")
        return 

    def save(self,fname):

        torch.nn.utils.vector_to_parameters(torch.tensor(self.W_flat_init), self.model.parameters())
        self.model.state_means = torch.from_numpy(self.state_mean)
        self.model.state_std = torch.from_numpy(self.state_std)
        torch.save(self.model.state_dict(), fname)
        
        n_ = os.path.basename(fname).split('.')[0] + f"_reward_Epi_{self.total_episode}.h5"
        b_name = os.path.join(os.path.dirname(fname),n_)

        # self.rankPrint(f"{b_name}")

        hf = h5py.File(b_name, 'w')
        hf.create_dataset('Reward',data=self.rolling_ep_rew_mean)
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
        