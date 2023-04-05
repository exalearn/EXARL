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
from turtle import forward
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
from ._build_torchMLP import MLPModel
from ._build_torchLSTM  import LSTMPolicyModel



from exarl.utils.globals import ExaGlobals
logger = ExaGlobals.setup_logger(__name__)
import h5py



# class MLPModel(nn.Module):

#     def __init__(self, in_size, out_size, num_l , l_size, 
#                  activ=nn.ReLU,
#                  out_activ=nn.Identity,
#                  squash_out = False):
#         super(MLPModel,self).__init__()

#         self.activation = activ()
#         self.out_activation = out_activ()
#         self.squash_out = squash_out
#         self.state_means = self.register_buffer(name="state_means",tensor=torch.zeros(in_size, requires_grad=False))
#         self.state_std = self.register_buffer(name="state_std",tensor=torch.ones(in_size, requires_grad=False))
        
#         if num_l == 0:
#             self.mlp_layers = []
#             self.mlp_out_layer = nn.Linear(in_size, out_size,bias=False)
#         else:
#             self.mlp_layers = nn.ModuleList([nn.Linear(in_size, l_size, bias=True)])
#             self.mlp_layers.extend([nn.Linear(l_size, l_size,bias=True) for _ in range(num_l-1)])
#             self.mlp_out_layer = nn.Linear(l_size, out_size,bias=True)

    
#     def forward(self,data):
#         # Normalize the input data with mean and std
#         data = (torch.as_tensor(data, dtype=self.state_means.dtype) - self.state_means) / self.state_std
       
#         for layer in self.mlp_layers:
#             data = self.activation(layer(data.float()))

#         # Last Layer of the network output..
#         out = self.out_activation(self.mlp_out_layer(data))

#         # Pass the output via tanh activation function
#         if self.squash_out:
#             out = nn.Tanh()(out)
#         return out

# class LSTMPolicyModel(nn.Module):

#     def __init__(self,in_size, out_size, num_l , l_size, 
#                  activ=nn.ReLU,
#                  out_activ=nn.Identity,
#                  squash_out = True):
#         super(LSTMPolicyModel, self).__init__()

#         self.activation = activ()
#         self.out_activation = out_activ()
#         self.squash_out = squash_out

#         self.l_size = l_size

#         self.state_means = self.register_buffer(name="state_means",tensor=torch.zeros(in_size, requires_grad=False))
#         self.state_std = self.register_buffer(name="state_std", tensor=torch.ones(in_size, requires_grad=False))

#         self.lstm = nn.LSTM(input_size=in_size, hidden_size=l_size,num_layers=num_l)
#         self.FC = nn.Linear(l_size, l_size,bias=True)
#         self.out_layer = nn.Linear(l_size, out_size,bias=True)
    
#     def forward(self,data):
#         # Normalize the input data with mean and std
#         data = (torch.as_tensor(data, dtype=self.state_means.dtype) - self.state_means) / self.state_std

#         out_ , (hn,cn) = self.lstm(data.float())

#         hn = hn.view(-1, self.l_size) #reshaping the data for Dense layer next
        
#         out = self.activation(hn)
#         out = self.activation(self.FC(out)) #first Dense
#         out = self.out_activation(self.out_layer(out)) #Final Output

#         # Pass the output via tanh activation function
#         if self.squash_out:
#             out = nn.Tanh()(out)
#         return out

        


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

        self.params = self.CreateParams()

        # Get the environnemt
        self.env = env
        if self.env is not None:
            self.env.seed(self.params['seed'])
        
        torch.manual_seed(self.params['seed'])
        # self.ob_dim = env.observation_space.shape[0]
        # self.ac_dim = env.action_space.shape[0]
        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.shape[0]

        

        # These parameter are used for delta setting.
        self.step_size= self.params['step_size']
        self.delta_std = self.params['delta_std']
        self.n_top = self.params['n_top']
        
        # if ExaComm.env_comm.size >1 :
        #     # divide the number if deltas amongs workers

        self.n_delta_in = self.params['n_delta']
        self.n_delta = None
        

        # This dummy, it is here just for the sake of workflow purpose.
        self.epsilon = ExaGlobals.lookup_params('epsilon')   
        self.n_steps = ExaGlobals.lookup_params('n_steps')

        # Create the Model
        self.model = self.CreateModel()
        self.rankPrint(self.model)
        
       
        # Fetch the Weights from the model... 
        W_torch = torch.nn.utils.parameters_to_vector(self.model.parameters())
        self.W_flat = W_torch.detach().numpy()
        
        # Initial Weights of the Model...
        self.W_flat_init =  W_torch.detach().numpy()
        # self.rankPrint(f"Initial Weight... {self.W_flat_init}")

        # Initialize the states mean and standard deviation.        
        self.state_means = np.zeros(self.ob_dim)
        self.state_std = np.ones(self.ob_dim)

        # Range function call from numpy random.
        self.rng = default_rng([self.params['seed'],self.agent_comm.rank])  
        self.deltas = None # This will be set when generate_PM_W is invoked.self

        # Storing list per step...reset will happen at the end of an episode.
        # in the generate data function...
        self.state_list = []
        self.act_list = []
        self.reward_list = []
        self.steps_complete = []
        self.steps_in = 0

        self.raw_rew_hist = []
        self.rolling_ep_rew_mean = []
        self.ModelEval_reward = []

        # Total_steps 
        self.total_steps = 0
        self.total_episode =0

        # Number of actors...
        self.Num_actors = self.agent_comm.size - self.agent_comm.num_learners


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
        param['squash_out'] = ExaGlobals.lookup_params('squash_out')
        param['num_l'] =  ExaGlobals.lookup_params('num_layers')
        param['l_size'] = size

        return param
        

    def CreateModel(self):
        # initialize policy
        if self.params['squash_out'] == 1:
            squ_out = True
        else:
            squ_out = False 

        if self.params['policy_type'] == 'linear':
            model = MLPModel(self.ob_dim, self.ac_dim, 
                            self.params['num_l'],
                            self.params['l_size'], 
                            activ=nn.Identity,
                            out_activ=nn.Identity,
                            squash_out= squ_out)
        elif self.params['policy_type'] == 'nonlinear':
            model = MLPModel(self.ob_dim, self.ac_dim, 
                            self.params['num_l'],
                            self.params['l_size'], 
                            activ=nn.ReLU,
                            out_activ=nn.Identity,
                            squash_out= squ_out)
        elif self.params['policy_type'] == 'LSTM':
            model = LSTMPolicyModel(self.ob_dim, self.ac_dim, 
                            self.params['num_l'],
                            self.params['l_size'], 
                            activ=nn.ReLU,
                            out_activ=nn.Identity,
                            squash_out= squ_out)							 																		
        else:
            raise NotImplementedError

        # Start with zero policy...
        # weights= torch.nn.utils.parameters_to_vector(model.parameters()).detach()
        # weights = torch.zeros_like(weights, requires_grad=False)
        # torch.nn.utils.vector_to_parameters(torch.tensor(weights,requires_grad=False), model.parameters())
    

        return model

    def Generate_pmW(self,W_flat_init):
        # Note:
        # The self.W_flat_init is coming from the agent instantiations.
        # The self.W_flat_init will be updated by the train on the call of learner
        # after the actor finishes the loop over all +/- weight.
        # self.rankPrint(f" Generate_pmW \n {self.W_flat_init} ")
        n_param = W_flat_init.shape[0]

        if self.Num_actors > 1 :
            if self.n_delta_in % self.Num_actors == 0:
                self.n_delta = int(self.n_delta_in / self.Num_actors)
            else:
                self.n_delta = int (np.ceil(self.n_delta_in / self.Num_actors))
                self.rankPrint(f"Setting the number of directions == {self.n_delta} on each worker.")
        else:
            self.n_delta = self.n_delta_in

        self.deltas = self.rng.standard_normal((self.n_delta, n_param))
        pm_W = np.concatenate((W_flat_init+(self.deltas*self.delta_std), W_flat_init-(self.deltas*self.delta_std)))
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
        
        # self.rankPrint(f"Set Weights... {(weights)}")
        # Set the model weights ...
        torch.nn.utils.vector_to_parameters(torch.tensor(weights,requires_grad=False), self.model.parameters())
        
        # Collect the set weights in the W_flat variable...
        # This will ensure that the get_weights call returns correct weights..
        W_torch = torch.nn.utils.parameters_to_vector(self.model.parameters())
        self.W_flat = W_torch.detach().numpy()
        
        
        self.model.state_means = torch.from_numpy(self.state_means)
        self.model.state_std = torch.from_numpy(self.state_std)

        self.model.float()
        # self.rankPrint(f" Weights-set from set weight:, {self.W_flat}")
        return
                           
    def action(self,state):
        # self.rankPrint(f"{state}")
        obs = np.asarray(state, dtype=np.float32)
        # self.rankPrint(f"{obs}, {type(obs)} {obs.reshape(1,-1)}......")
        obs = torch.from_numpy(obs.reshape(1,-1)).float()
        # self.rankPrint(f"{self.model(obs).detach()} Inaction...")
        
        act = self.model(obs)
        if act.numel() == 1:
            act = act.detach().numpy()
        else:
            act = act.squeeze().detach().numpy()
        # .numpy()
        # act = np.array(act,dtype=np.float32)
        # self.rankPrint(f"{act}, {type(act)}")
        return act , self.params['policy_type']

    def BatchCheck_PreProcessing(self, batch):
        if isinstance(batch, list):
            batch_new = {}
            Rollout_ep_rew_mean = []
            Top_returns = [] 
            P_returns = np.array([])
            M_returns = np.array([])

            for b in batch:
                Rollout_ep_rew_mean.append(b['rollout/ep_rew_mean']) # np 
                Top_returns += b['top_returns'] # list
                P_returns = np.append ( P_returns, b['p_returns'] ) # np array
                M_returns = np.append ( M_returns, b['p_returns'] )
                
            # Collect all the information in new dict.
            batch_new['deltas'] = np.concatenate([ b['deltas'] for b in batch])
            batch_new['states'] = np.concatenate([ b['states'] for b in batch])
            batch_new['p_returns'] = P_returns
            batch_new['m_returns'] = M_returns
            batch_new['top_returns'] = Top_returns
            batch_new['rollout/ep_rew_mean'] = Rollout_ep_rew_mean

            return batch_new
        else:
            return batch

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

        # if isinstance(batch,list):
        #     for b in batch:


        # else:

        batch = self.BatchCheck_PreProcessing(batch)

        rollout_ep_rew_mean = batch['rollout/ep_rew_mean'] # np 
        top_returns = batch['top_returns'] # list
        p_returns = batch['p_returns'] # np array
        m_returns = batch['m_returns'] # np array
        states = batch['states'] # np array
        deltas = batch['deltas'] # 

        # self.rankPrint(f"{deltas.shape}.... >>")
        # self.rankPrint(f"{p_returns.shape}.... >>")
        # self.rankPrint(f"Epi-count - {self.total_episode} top_returns; \n {top_returns}")
        
        top_idx = sorted(range(len(top_returns)), key=lambda k: top_returns[k], reverse=True)[:self.n_top]
        p_returns = np.stack(p_returns)[top_idx]
        m_returns = np.stack(m_returns)[top_idx]

        return_std = np.concatenate((p_returns, m_returns)).std() + 1e-6
        step_size = self.step_size / (self.n_top * return_std + 1e-6)

        # if self.total_episode % 10 == 0:
            # self.rankPrint(f"Episode {self.total_episode} rollout/ep_rew_mean :: {rollout_ep_rew_mean}")
            # self.rankPrint(f"Episode={self.total_episode} Top Returns Mean :: {np.stack(top_returns)[top_idx].mean()} ")
            # self.rankPrint(f"Step Size :: {step_size}, original :: {self.step_size}")
            # self.rankPrint(f"+/- reward Mean : {(p_returns.mean() + m_returns.mean())/2}")
            # print(f"{self.total_episode} : mean return: {top_returns.mean()}, top_return: {top_returns.max()}")
        
        # self.rankPrint(f"Top return mean :: {np.stack(top_returns)[top_idx].mean()}")
        self.raw_rew_hist.append(np.stack(top_returns)[top_idx].mean())
        self.rolling_ep_rew_mean.append(rollout_ep_rew_mean)


        p_2 = np.sum((p_returns - m_returns)*deltas[top_idx].T, axis=1)
        
        
        # self.rankPrint(f" Train....W_flat_init before update \n {self.W_flat_init} ")
        
        
        # Update the weights based on the ARS differencing scheme. 
        self.W_flat_init = self.W_flat_init +  step_size * p_2

        # self.rankPrint(f" Train....W_flat_init after update \n {self.W_flat_init} ")

        ep_steps = states.shape[0]
        # self.rankPrint(f"{ep_steps}, {states.shape},,, >>>")
        self.state_means = self.Mean_update(states, self.state_means, self.total_steps)
        self.state_std = self.Std_update(states, self.state_std, self.total_steps)

        self.total_steps += ep_steps
        self.total_episode += 1
        self.delta_std *= self.params["decay"]

        return np.stack(top_returns)[top_idx].mean(), None, None

    def remember(self, state, action, reward, next_state,steps, done):
        # self.rankPrint(f">>>><<< {len(reward)}, {reward}")
        # self.rankPrint(f" >>> {len(state)},  {type(state)}, {len(action)}, {type(action)}, {len(reward)}, {type(reward)}")
        
        # self.rankPrint(f"{len(state)}, {state}, {type(state)}")
        state = state.reshape(self.ob_dim,-1)

        self.state_list.append(state)
        self.act_list.append(action)
        self.reward_list.append(reward)

        # if the environment return termination i.e True
        if done:
            self.steps_in += 1
            # self.rankPrint(f"{self.steps_in}, ***")
            # append the number of steps completed
            self.steps_complete.append(self.steps_in)
            # Reset the counter.    
            self.steps_in = 0
        else:
            self.steps_in += 1
            # self.rankPrint(f"{self.steps_in}, ***")
            
        return

    def CollectEpisodeReward(self,CumSum,rewardlist):
        k = 0
        reward = []
        i_ = 0
        for i in range(len(CumSum)):
            # print(i,i_,CumSum[i])
            reward.append(np.sum(rewardlist[i_:CumSum[i]]))
            i_ = CumSum[i]
        
        return np.array(reward)

    def generate_data(self):
        batch_data = {}

        # This is a condition useful in case the env terminate early without completing
        # all the steps.
        # print(np.array(self.steps_complete), self.n_steps)

        if np.any(np.array(self.steps_complete) < self.n_steps):
            self.rankPrint(f"Few runs did? not finished total number of steps..")
            # self.rankPrint(f"Number of perturbations run:{len(self.steps_complete)}")
            # self.rankPrint(f"{np.array(self.reward_list).shape}")
            
            # This is a array with a list of steps completed in each direction.
            sp = np.cumsum(np.array(self.steps_complete))

            # self.rankPrint(sp)
            rewardSum_perEpisode = self.CollectEpisodeReward(sp,np.array(self.reward_list) )

            # # Split the array according to the number of steps completed for each direction
            # t_ = np.split(np.array(self.reward_list), sp)
            
            # self.rankPrint(f"{(t_)}")
            # tmp_ = np.array([ k.shape for k in t_ ])
            # self.rankPrint(f"{tmp_}")
        
            # Calculate the sum of rewards collected at each step in an episode.
            # rewardSum_perEpisode = np.array([ np.sum(k)for k in t_ ]) 
        else:
            self.reward_list = np.array(self.reward_list).reshape(self.n_delta*2, self.n_steps)

            # sum over all the n_steps in a single episode run
            rewardSum_perEpisode = np.sum(self.reward_list,axis=1)
        
        
        # self.rankPrint(f"{len(rewardSum_perEpisode)},>>>>")
        


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

        top_idx = sorted(range(len(top_returns)), key=lambda k: top_returns[k], reverse=True)[:self.n_top]
        
        # This is mean of all +/-ve rollout 
        batch_data['rollout/ep_rew_mean'] = np.mean(rewardSum_perEpisode)
        batch_data['top_returns'] = top_returns
        batch_data['p_returns'] = rewardSum_perEpisode[:self.n_delta]
        batch_data['m_returns'] = rewardSum_perEpisode[self.n_delta:]
        batch_data['states'] = np.stack(self.state_list).squeeze()
        batch_data['act_arr'] = np.stack(self.act_list)
        batch_data['deltas'] = self.deltas
        batch_data['top_idx'] = top_idx
        
        # Reset the storing list
        self.state_list = []
        self.act_list = []
        self.reward_list = []
        self.steps_complete = []
        self.steps_in = 0
        
        # Update the mean and sta
        yield batch_data
    
    def target_train(self):
        return self.model.parameters()

    def update_target(self):
        pass
        # print("Implement update method in ARS.py")
        return 

    def load(self,weight_file):
        self.rankPrint("Load method called to Load the weights in ARS.py")
        
        # This step will create appropriate regitery buffer for mean and std
        # to be loaded directly from the file.        
        r1,r2 = np.random.random(self.ob_dim), np.random.random(self.ob_dim)
        self.model.state_means = torch.from_numpy(r1)
        self.model.state_std = torch.from_numpy(r2)
        
        # Load the state_dict ....
        self.model.load_state_dict(torch.load(weight_file))

        # Extract the weights....
        W_torch = torch.nn.utils.parameters_to_vector(self.model.parameters())
        
        # Loaded trained initial weights for the Model...
        self.W_flat_init =  W_torch.detach().numpy()
      
        return 

    def save(self,fname):

        # self.rankPrint(f"{self.W_flat_init}.. Saving weights..")
        torch.nn.utils.vector_to_parameters(torch.tensor(self.W_flat_init), self.model.parameters())
        self.model.state_means = torch.from_numpy(self.state_means)
        self.model.state_std = torch.from_numpy(self.state_std)

        # self.rankPrint(f"saving mean {self.model.state_means}")
        # self.rankPrint(f"saving std {self.model.state_std }")

        f_ =  os.path.basename(fname).split('.')[0] + ".pth"
        f_name = os.path.join(os.path.dirname(fname), f_)

        # self.rankPrint("Torch saving... The state Dict...")
        torch.save(self.model.state_dict(), f_name)
        # torch.save(self.model, f_name)
        
        n_ = os.path.basename(fname).split('.')[0] + f"_rewards.h5"
        b_name = os.path.join(os.path.dirname(fname),n_)

        hf = h5py.File(b_name, 'w')
        hf.create_dataset('rollout_ep_rew_mean',data=self.rolling_ep_rew_mean)
        hf.create_dataset('TopReward_mean',data=self.raw_rew_hist)
        hf.create_dataset('Modelevaluation_reward', data=self.ModelEval_reward)

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
        