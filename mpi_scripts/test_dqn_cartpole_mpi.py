# -*- coding: utf-8 -*-
import random, math
import gym
from tqdm import tqdm
import time

##
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RL-Logger')
logger.setLevel(logging.ERROR)

from agents.dqn import DQN

import csv

if __name__ == "__main__":
    import sys

    #########
    ## MPI ##
    #########
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    ###########
    ## Train ##
    ###########
    EPISODES = 2000
    NSTEPS   = 200

    #######################
    ## Setup environment ##
    #######################
    estart = time.time()
    env = gym.make('CartPole-v0')
    env._max_episode_steps=NSTEPS
    env.seed(1)
    end = time.time()
    logger.info('Time init environment: %s' % str( (end - estart)/60.0))
    logger.info('Using environment: %s' % env)
    logger.info('Observation_space: %s' % env.observation_space.shape)
    logger.info('Action_size: %s' % env.action_space)

    #################
    ## Setup agent ##
    #################
    agent = DQN(env)
    search_type= 'epsilon'
    
    for e in tqdm(range(EPISODES), desc='RL Episodes', leave=True):
        logger.info('Starting new episode: %s' % str(e))
        current_state = env.reset()
        total_reward=0
        done = False
        while done!=True:
            action = agent.action(current_state)
            next_state, reward, done, _ = env.step(action)
            output  = env.step(action)
            ## SHOULD GO TO THE LEARNER ##
            agent.remember(current_state, action, reward, next_state, done)
            if rank == 0:
                print('### Train ###')
                agent.train()
                target_weights = agent.target_weights
                agent.target_weights = comm.bcast(target_weights, root=0)
                
            logger.info('Rank[%s] - Current state: %s' % (str(rank),str(current_state)))
            logger.info('Rank[%s] - Action: %s' % (str(rank),str(action)))
            logger.info('Rank[%s] - Next state: %s' % (str(rank),str(next_state)))
            logger.info('Rank[%s] - Reward: %s' % (str(rank),str(reward)))
            logger.info('Rank[%s] - Done: %s' % (str(rank),str(done)))
            
            ##
            current_state = next_state
            ##
            total_reward+=reward
            logger.info('Rank[%s] - Current episode reward: %s ' % (str(rank),str(total_reward)))

    ## Save Learning target model
    if rank==0:
        agent.save(filename_prefix+'.h5')

