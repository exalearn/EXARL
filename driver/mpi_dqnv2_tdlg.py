# -*- coding: utf-8 -*-
import mpi4py.rc; mpi4py.rc.threads = False
from mpi4py import MPI
import random, math
import gym
import time

##
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RL-Logger')
logger.setLevel(logging.INFO)

#import exarl as erl

import csv
import sys

from agents.agent_vault.dqn_v2 import DQN

if __name__ == "__main__":

    import time
    start = time.time()
    
    #########
    ## MPI ##
    #########
    comm = MPI.COMM_WORLD
    rank = comm.rank#Get_rank()
    size = comm.size#Get_size()
 
    ###########
    ## Train ##
    ###########
    EPISODES = 2500
    NSTEPS   = 10

    #######################
    ## Setup environment ##
    #######################
    #estart = time.time()
    env = gym.make('envs:TDLG-v0')
    env._max_episode_steps=NSTEPS
    env.seed(1)
    #end = time.time()
    #logger.info('Time init environment: %s' % str( (end - estart)/60.0))
    #logger.info('Using environment: %s' % env)
    #logger.info('Observation_space: %s' % env.observation_space.shape)
    #logger.info('Action_size: %s' % env.action_space)

    #################
    ## Setup agent ##
    #################
    agent = DQN(env,cfg='agents/agent_vault/agent_cfg/dqn_setup.json')
    ## Save infomation ##
    filename_prefix = 'dqn_cartpole_msle_episode%s_memory%s_rank%s_v1' % (str(EPISODES),str(NSTEPS),str(rank))
    train_file = open(filename_prefix + ".log", 'w')
    train_writer = csv.writer(train_file, delimiter = " ")

    target_weights = None 
    rank0_memories = 0
    
    #for e in tqdm(range(EPISODES), desc='RL Episodes', leave=True):
    for e in (range(EPISODES)):
        ##comm.barrier()
        #print('Rank[%s] - Starting new episode: %s' % (str(rank),str(e)))
        current_state = env.reset()
        total_reward = 0
        done = False
        all_done = False
        
        start_time_episode = time.time()
        steps = 0
        while all_done != True:
            ## All workers ##
            if done != True:
                ## All workers ##
                action = agent.action(current_state)
                next_state, reward, done, _ = env.step(action)
                total_reward+=reward
                memory = (current_state, action, reward, next_state, done, total_reward)
                
            new_data = comm.gather(memory, root=0)
            #logger.info('Rank[%s] - Memory length: %s ' % (str(rank),len(agent.memory)))

            ## Learner ##
            if comm.rank==0:
                ## Push memories to learner ##
                #[agent.remember(data[0],data[1],data[2],data[3],data[4]) for data in new_data]
                    
                for data in new_data:
                    agent.remember(data[0],data[1],data[2],data[3],data[4])
                ## Train learner ##
                agent.train()
                rank0_memories = len(agent.memory)
                target_weights = agent.target_model.get_weights()

            ## Broadcast the memory size and the model weights to the workers  ##
            rank0_memories = comm.bcast(rank0_memories, root=0)
            current_weights = comm.bcast(target_weights, root=0)
            #logger.info('Rank[%s] - rank0 memories: %s' % (str(rank),str(rank0_memories)))

            ## Set the model weight for all the workers
            if comm.rank>0 and rank0_memories%(size*5)==0:
            #    #logger.info('## Rank[%s] - Updating weights ##' % str(rank))
                agent.target_model.set_weights(current_weights)

            ## Print
            #logger.info('Rank[%s] - Current state: %s' % (str(rank),str(current_state)))
            #logger.info('Rank[%s] - Action: %s' % (str(rank),str(action)))
            #logger.info('Rank[%s] - Next state: %s' % (str(rank),str(next_state)))
            #logger.info('Rank[%s] - Reward: %s' % (str(rank),str(reward)))
            #logger.info('Rank[%s] - Done: %s' % (str(rank),str(done)))
            
            ## Update state
            current_state = next_state

            ## Save memory for offline analysis
            train_writer.writerow([current_state,action,reward,next_state,total_reward,done])
            train_file.flush()
                    
            ## Print current cumulative reward
            #logger.info('Rank[%s] - Current episode reward: %s ' % (str(rank),str(total_reward)))
            steps += 1
            if steps >= NSTEPS:
                done = True
            all_done = comm.allreduce(done, op=MPI.LAND)

    elapse = time.time() -start        
    sum_elapse = comm.reduce(elapse, op=MPI.SUM, root=0)
    ## Save Learning target model
    if comm.rank==0:
        agent.save(filename_prefix+'.h5')
        train_file.close()
        print(" Average elapsed time = ", sum_elapse/size)
