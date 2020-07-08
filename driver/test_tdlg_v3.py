# -*- coding: utf-8 -*-
import mpi4py.rc; mpi4py.rc.threads = False
from mpi4py import MPI
import random, math
import gym
import time
import numpy as np

##
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RL-Logger')
logger.setLevel(logging.INFO)

#import exarl as erl

import csv
import sys

import time
import gym
import envs
import agents

from agents.agent_vault.dqn_lstm import DQN_LSTM

if __name__ == "__main__":

    import time
    start = time.time()
    
    #########
    ## MPI ##
    #########
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
 
    ###########
    ## Train ##
    ###########
    EPISODES = 1
    NSTEPS   = 10

    #######################
    ## Setup environment ##
    #######################
    env = gym.make('envs:ExaLearnBlockCoPolymerTDLG-v3', env_comm=comm)
    agent = agents.make('DQN-LSTM-v0', env=env, agent_comm=comm)
    logger.info('Using environment: %s' % env)
    print('Observation_space: %s' % env.observation_space.shape)
    logger.info('Action_size: %s' % env.action_space)
    current_state = env.reset()
    total_reward=0
    nsteps=0
    estart = time.time()
    for i in range(NSTEPS):
        nsteps+=1
        action = np.array(1)
        next_state, reward, done, _ = env.step(action)
        total_reward+=reward
        print('Step #{} action/reward/total_reward/done: {}/{}/{}/{}'.format(i,action,reward,total_reward,done))
        if done:
            break
    end = time.time()
    ave_step_time=((end - estart))/nsteps
    print('Average time per step: {}'.format(ave_step_time))

    total_reward = round(total_reward,6)
    target_reward = round(-0.29159110028893626,6)
    result_diff  = abs(total_reward-target_reward)
    print('Results difference: {}'.format(result_diff))
    if  result_diff<0.0001:
        sys.exit('Test Passed => total reward: {}'.format(total_reward))
    else:
        sys.exit('Test Failed => total reward: {}'.format(total_reward))
