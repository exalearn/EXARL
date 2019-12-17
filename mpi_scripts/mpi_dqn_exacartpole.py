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

import agents

import csv

if __name__ == "__main__":
    import sys

    #########
    ## MPI ##
    #########
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    logger.info("Rank: %s" % rank)
    logger.info("Size: %s" % size)
    
    ###########
    ## Train ##
    ###########
    EPISODES = 200
    NSTEPS   = 200

    #######################
    ## Setup environment ##
    #######################
    estart = time.time()
    env = gym.make('exa_gym:ExaLearnCartpole-v0')
    env._max_episode_steps=NSTEPS
    #env.seed(1)
    end = time.time()
    logger.info('Time init environment: %s' % str( (end - estart)/60.0))
    logger.info('Using environment: %s' % env)
    logger.info('Observation_space: %s' % env.observation_space.shape)
    logger.info('Action_size: %s' % env.action_space)

    #################
    ## Setup agent ##
    #################
    agent = agents.DQN(env)
    search_type= 'epsilon'

    ##################
    ## Save results ##
    ##################
    filename_prefix = 'dqn_exacartpole_'+search_type+'_episode%s_memory%s_rank%s_v1' % (str(EPISODES),str(NSTEPS),str(rank))
    train_file = open(filename_prefix + ".log", 'w')
    train_writer = csv.writer(train_file, delimiter = " ")

    target_weights = None 
    rank0_memories = 0
    
    for e in tqdm(range(EPISODES), desc='RL Episodes', leave=True):
        logger.info('Starting new episode: %s' % str(e))
        current_state = env.reset()
        total_reward=0
        done = False
        while done!=True:

            ## All workers ##
            action = agent.action(current_state)
            next_state, reward, done, _ = env.step(action)
            total_reward+=reward
            memory = (current_state, action, reward, next_state, done, total_reward)
            new_data = comm.gather(memory, root=0)
            logger.info('Rank[%s] - Memory length: %s ' % (str(rank),len(agent.memory)))

            ## Learner ##
            if comm.rank==0:
                ## Push memories to learner ##
                for data in new_data:
                    agent.remember(data[0],data[1],data[2],data[3],data[4])
                    
                ## Train learner ##
                agent.train()
                rank0_memories = len(agent.memory)
                target_weights = agent.target_model.get_weights()

            ## Broadcast the memory size and the model weights to the workers  ##
            rank0_memories = comm.bcast(rank0_memories, root=0)
            current_weights = comm.bcast(target_weights, root=0)
            logger.info('Rank[%s] - rank0 memories: %s' % (str(rank),str(rank0_memories)))

            ## Set the model weight for all the workers
            if comm.rank>0 and rank0_memories%(size*5)==0:
                logger.info('## Rank[%s] - Updating weights ##' % str(rank))
                agent.target_model.set_weights(current_weights)

            ## Print
            logger.info('Rank[%s] - Current state: %s' % (str(rank),str(current_state)))
            logger.info('Rank[%s] - Action: %s' % (str(rank),str(action)))
            logger.info('Rank[%s] - Next state: %s' % (str(rank),str(next_state)))
            logger.info('Rank[%s] - Reward: %s' % (str(rank),str(reward)))
            logger.info('Rank[%s] - Done: %s' % (str(rank),str(done)))
            
            ## Update state
            current_state = next_state

            ## Save memory for offline analysis
            train_writer.writerow([current_state,action,reward,next_state,total_reward,done])
            train_file.flush()
                    
            ## Print current cumulative reward
            logger.info('Rank[%s] - Current episode reward: %s ' % (str(rank),str(total_reward)))
                
    ## Save Learning target model
    if comm.rank==0:
        agent.save(filename_prefix+'.h5')
        train_file.close()

