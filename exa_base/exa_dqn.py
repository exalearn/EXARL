# -*- coding: utf-8 -*-
import gym
import exa_envs
import exa_agents

#
import os
import csv

#
from mpi4py import MPI

##
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RL-Logger')
logger.setLevel(logging.INFO)

class ExaDQN:
    def __init__(self, agent_id, env_id):
        ## Default training 
        self.nepisodes=1
        self.nsteps=10
        self.results_dir='./results/'
        self.do_render=False
        
        ## Setup agent and environents
        self.agent_id = agent_id
        self.env_id   = env_id
        self.env = gym.make(env_id)
        self.env._max_episode_steps=self.nsteps
        self.agent = exa_agents.make(agent_id, env=self.env)
        
    def set_training(self,nepisodes,nsteps):
        self.nepisodes = nepisodes
        self.nsteps    = nsteps
        self.env._max_episode_steps=self.nsteps

    def set_results_dir(self,results_dir):
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        ## General 
        self.results_dir=results_dir
        ## Set for agent
        ## Set for env

    def render_env(self):
        self.do_render=True
        
    def run(self):
        #########
        ## MPI ##
        #########
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        ##
        rank0_memories = 0
        target_weights = None

        ##################
        ## Save results ##
        ##################
        filename_prefix = 'ExaDQN_' + 'Episode%s_Steps%s_Rank%s_memory_v1' % ( str(self.nepisodes), str(self.nsteps), str(rank))
        train_file = open(self.results_dir+'/'+filename_prefix + ".log", 'w')
        train_writer = csv.writer(train_file, delimiter = " ")

        ## For Environments ##
        self.env.set_results_dir(self.results_dir+'/rank'+str(rank))
        #if self.render_env: self.env.render()

        for e in range(self.nepisodes):
            current_state = self.env.reset()
            total_reward=0
            for s in range(self.nsteps):
                print('Rank[%s] - Episode/Step %s/%s' % (str(rank),str(e),str(s)))
                
                ## All workers ##
                action = self.agent.action(current_state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward+=reward
                memory = (current_state, action, reward, next_state, done, total_reward)
                new_data = comm.gather(memory, root=0)
                print('Rank[%s] - Memory length: %s ' % (str(rank),len(self.agent.memory)))

                ## Learner ##
                if comm.rank==0:
                    ## Push memories to learner ##
                    for data in new_data:
                        self.agent.remember(data[0],data[1],data[2],data[3],data[4])
                    
                        ## Train learner ##
                        self.agent.train()
                        rank0_memories = len(self.agent.memory)
                        target_weights = self.agent.target_model.get_weights()

                ## Broadcast the memory size and the model weights to the workers  ##
                rank0_memories = comm.bcast(rank0_memories, root=0)
                current_weights = comm.bcast(target_weights, root=0)
                print('Rank[%s] - rank0 memories: %s' % (str(rank),str(rank0_memories)))

                ## Set the model weight for all the workers
                if comm.rank>0 and rank0_memories%(size*5)==0:
                    print('## Rank[%s] - Updating weights ##' % str(rank))
                    self.agent.target_model.set_weights(current_weights)

                ## Update state
                current_state = next_state
                print('Rank[%s] - Total Reward:%s' % (str(rank),str(total_reward) ))
                      
                ## Save memory for offline analysis
                train_writer.writerow([current_state,action,reward,next_state,total_reward,done])
                train_file.flush()

        ## Save Learning target model
        if comm.rank==0:
            self.agent.save(self.results_dir+filename_prefix+'.h5')
            train_file.close()
