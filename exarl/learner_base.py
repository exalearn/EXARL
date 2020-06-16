# © (or copyright) 2020. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
# National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
# Department of Energy/National Nuclear Security Administration. All rights in the program are
# reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
#1;95;0c Security Administration. The Government is granted for itself and others acting on its behalf a
# nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
# derivative works, distribute copies to the public, perform publicly and display publicly, and
# to permit others to do so.


import time
import gym
import envs
import agents

import os
import csv

from mpi4py import MPI
import json

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RL-Logger')
logger.setLevel(logging.ERROR )

class ExaLearner():

    def __init__(self, run_params):
        # Default training
        self.nepisodes = 1
        self.nsteps = 10
        self.results_dir = './results' # Default dir, will be overridden by candle 
        self.do_render = False

        self.run_type = run_params['run_type']
        self.mpi_children_per_parent = int(run_params['mpi_children_per_parent'])
        # Setup agent and environments
        agent_id = 'agents:'+run_params['agent']
        env_id   = 'envs:'+run_params['env']
        self.agent_id = agent_id
        self.env_id   = env_id
        self.agent, self.env = self.make()
        #self.env._max_episode_steps = self.nsteps
        self.env.spec.max_episode_steps  = self.nsteps

        self.leader = False
        self.worker_begin = -1

        # Set configuration
        self.mpi_children_per_parent = run_params['mpi_children_per_parent']
        self.set_config(run_params)

    def make(self):
        # World communicator
        self.world_comm = MPI.COMM_WORLD
        world_rank = self.world_comm.rank
        world_size = self.world_comm.size

        if self.run_type == 'static-two-groups':
            ### leaders: {0:worker_begin-1}, workers: {worker_begin, size-1}
            self.worker_begin = int(world_size / self.mpi_children_per_parent)
            if self.worker_begin == 0:
                print('[Aborting] Worker and Leader cannot have the same rank. Increase #processes and try again.')
                self.world_comm.Abort()
            if world_rank < self.worker_begin:
                self.leader = True 
            ### Identify leader colors (assumes 0 is *always* a leader)
            color = 2
            if world_rank >= self.worker_begin:
                color = 1
            ### leaders(2) and workers(1) intracomm
            self.intracomm = self.world_comm.Split(color, world_rank)
            env = gym.make(self.env_id, env_comm=self.intracomm)
            # group 1 (worker) communicates with group 2 (leader)
            if color == 1:
                self.intercomm = MPI.Intracomm.Create_intercomm(self.intracomm, 0, self.world_comm, 0)
            # group 2 (leader) communicates with group 1 (worker)
            agent = None
            if color == 2:
                self.intercomm = MPI.Intracomm.Create_intercomm(self.intracomm, 0, self.world_comm, self.worker_begin)
                agent = agents.make(self.agent_id, env=env, agent_comm=self.intracomm)
        elif self.run_type == 'static-multi-groups':
            ### Assumes 0 is *always* the leader of agents
            ncolors = self.mpi_children_per_parent+1
            color = int(world_rank % ncolors)
            if color == 0: 
                self.leader = True
            # one-to-many group communication
            self.intracomm = self.world_comm.Split(color, world_rank)
            env = gym.make(self.env_id, env_comm=self.intracomm)
            self.intercomm = [MPI.COMM_NULL]*(ncolors-1)
            agent = None
            if color == 0:
                agent = agents.make(self.agent_id, env=env, agent_comm=self.intracomm)
                for i in range(ncolors-1):
                    self.intercomm[i] = MPI.Intracomm.Create_intercomm(self.intracomm, 0, self.world_comm, i+1)
            else:
                self.intercomm[0] = MPI.Intracomm.Create_intercomm(self.intracomm, 0, self.world_comm, 0)
        else:
            # Environment communicator
            env_color = int(world_rank/(self.mpi_children_per_parent))#+1))
            self.env_comm = self.world_comm.Split(env_color, world_rank)

            # Create environment object
            env = gym.make(self.env_id, env_comm=self.env_comm)

            # Agent communicator
            agent_color = MPI.UNDEFINED
            if world_rank%(self.mpi_children_per_parent+1) == 0:
                agent_color = 0 # Can be anything, just assigning a common value for color
            self.agent_comm = self.world_comm.Split(agent_color, world_rank)
            # Create agent object
            agent = None
            if world_rank%(self.mpi_children_per_parent+1) == 0:
                agent = agents.make(self.agent_id, env=env, agent_comm=self.agent_comm)

        return agent, env


    def set_training(self,nepisodes,nsteps):
        self.nepisodes = nepisodes
        self.nsteps    = nsteps
        #self.env._max_episode_steps = self.nsteps
        self.env.spec.max_episode_steps  = self.nsteps

    # Use with CANDLE
    def set_config(self, params):
        self.set_training(int(params['n_episodes']), int(params['n_steps']))
        # set the agent up
        self.agent.set_config(params)
        self.env.set_config(params)
        self.results_dir = params['output_dir']
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def render_env(self):
        self.do_render=True

    def run_exarl(self, comm):

        filename_prefix = 'ExaLearner_' + 'Episodes%s_Steps%s_Rank%s_memory_v1' % ( str(self.nepisodes), str(self.nsteps), str(comm.rank))
        train_file = open(self.results_dir+'/'+filename_prefix + ".log", 'w')
        train_writer = csv.writer(train_file, delimiter = " ")
        #print('self.world_comm.rank:',self.world_comm.rank)

        for e in range(self.nepisodes):

            rank0_memories = 0
            target_weights = None
            current_state = self.env.reset()
            total_reward = 0
            done = False
            all_done = False

            start_time_episode = time.time()
            steps = 0
            while all_done != True:
                ## All workers ##
                if done != True:
                    action = self.agent.action(current_state)
                    next_state, reward, done, _ = self.env.step(action)
                    total_reward += reward
                    memory = (current_state, action, reward, next_state, done, total_reward)

                new_data = comm.gather(memory, root=0)
                logger.info('Rank[%s] - Memory length: %s ' % (str(comm.rank),len(self.agent.memory)))

                ## Learner ##
                if comm.rank == 0:
                    ## Push memories to learner ##
                    for data in new_data:
                        self.agent.remember(data[0],data[1],data[2],data[3],data[4])
                        ## Train learner ##
                        self.agent.train()
                        rank0_memories = len(self.agent.memory)
                        target_weights = self.agent.get_weights()
                        if rank0_memories%(comm.size) == 0:
                            self.agent.save(self.results_dir+'/'+filename_prefix+'.h5')

                ## Broadcast the memory size and the model weights to the workers  ##
                rank0_memories = comm.bcast(rank0_memories, root=0)
                current_weights = comm.bcast(target_weights, root=0)

                logger.info('Rank[%s] - rank0 memories: %s' % (str(comm.rank), str(rank0_memories)))

                ## Set the model weight for all the workers
                if comm.rank > 0 and rank0_memories > 30:# and rank0_memories%(size)==0:
                    logger.info('## Rank[%s] - Updating weights ##' % str(comm.rank))
                    self.agent.set_weights(current_weights)

                ## Update state
                current_state = next_state
                logger.info('Rank[%s] - Total Reward:%s' % (str(comm.rank),str(total_reward)))

                ## Save memory for offline analysis
                train_writer.writerow([current_state,action,reward,next_state,total_reward,done])
                train_file.flush()

                ## Save Learning target model
                if comm.rank == 0:
                    self.agent.save(self.results_dir+'/'+filename_prefix+'.h5')

                steps += 1
                if steps >= self.nsteps:
                    done = True

                all_done = comm.allreduce(done, op=MPI.LAND)

            end_time_episode = time.time()
            logger.info('Rank[%s] run-time for episode %s: %s ' % (str(comm.rank), str(e), str(end_time_episode - start_time_episode)))

        train_file.close()
        
    ### Uses two intercomms for communicating agent comm with environment comms
    def run_exarl_two_groups(self, intracomm, intercomm):
        rank0_memories = 0
        target_weights = None
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        if self.leader is False: # only workers will update
            filename_prefix = 'ExaLearner_' + 'Episodes%s_Steps%s_Rank%s_memory_v1' % ( str(self.nepisodes), str(self.nsteps), str(comm.rank))
            train_file = open(self.results_dir+'/'+filename_prefix + ".log", 'w')
            train_writer = csv.writer(train_file, delimiter = " ")

        start_time_episode = time.time()
        steps = 0
        for e in range(self.nepisodes):
            current_state = self.env.reset()
            total_reward=0
            done = True # for leaders
            if self.leader is False:
                done = False
            all_done = False
            root = 0
            if rank == 0:
                root = MPI.ROOT # remote leader
            if rank != 0 and self.leader is True:
                root = MPI.PROC_NULL
 
            while all_done!=True:

                worker_state = None
                new_data = [] 

                ### workers
                if self.leader is False:
                    done = intracomm.allreduce(done, op=MPI.LAND)
                    if done != True:
                        action = self.agent.action(current_state)
                        next_state, reward, done, _ = self.env.step(action)
                        total_reward += reward
                        worker_state = (action, reward, next_state, done, total_reward)

                ### communicate from workers to remote leader of workers                   
                worker_state = intercomm.gather(worker_state, root=root)
                
                ## Learner (also a leader) ##
                if rank == 0:
                    for wdata in worker_state:
                        if wdata is not None:
                            new_data.append([current_state, wdata[0], wdata[1], wdata[2], wdata[3], wdata[4]])

                    ## Push memories to learner ##
                    if new_data is not None:
                        for data in new_data:
                            self.agent.remember(data[0],data[1],data[2],data[3],data[4])
                    
                        ## Train learner ##
                        self.agent.train()
                        rank0_memories = len(self.agent.memory)
                        target_weights = self.agent.get_weights()
                    
                    if rank0_memories%(intracomm.size) == 0:
                        self.agent.save(self.results_dir+'/'+filename_prefix+'.h5')

  
                ### communicate from remote leader to local leader of workers
                ## Broadcast the memory size and the model weights to the workers  ##        
                ## TODO only send to worker root
                rank0_memories = intercomm.bcast(rank0_memories, root=root)
                current_weights = intercomm.bcast(target_weights, root=root)
                new_data = intercomm.bcast(new_data, root=root)
                
                ## Set the model weight for all the workers
                if self.leader is False:
                    if rank0_memories is not None and rank0_memories>30:                            
                        self.agent.set_weights(current_weights)

                    ## Update state
                    if done != True:
                        current_state = next_state
                    
                    ## Save memory for offline analysis
                    # current_state,action,reward,next_state,total_reward,done
                    if new_data is not None:
                        for data in new_data:
                            train_writer.writerow([current_state,data[1],data[2],data[3],data[5],data[4]])
                            train_file.flush()
                           
                ## Save Learning target model
                if rank == 0:
                    self.agent.save(self.results_dir+'/'+filename_prefix+'.h5')

                steps += 1
                if steps >= self.nsteps:
                    done = True

                ## Exit criteria
                all_done = comm.allreduce(done, op=MPI.LAND)
            
            end_time_episode = time.time()
            mtim = end_time_episode - start_time_episode 
            ptim = comm.reduce(mtim, op=MPI.SUM, root=0)
            if rank == 0:
                logger.info('Average execution time (in secs.) for %s episodes: %s ' % (str(rank), str(e), str(ptim / size)))

            if self.leader is False:
                train_file.close()
 
    ### Uses multiple intercomms for communicating agent comm with environment comms
    def run_exarl_multi_groups(self, intracomm, intercomm):
        rank0_memories = 0
        target_weights = None
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        ## TODO do not compute again
        ncolors = int(self.mpi_children_per_parent)+1
        color = int(rank % ncolors)
        
        if self.leader is False: # only workers will update
            filename_prefix = 'ExaLearner_' + 'Episodes%s_Steps%s_Rank%s_memory_v1' % ( str(self.nepisodes), str(self.nsteps), str(comm.rank))
            train_file = open(self.results_dir+'/'+filename_prefix + ".log", 'w')
            train_writer = csv.writer(train_file, delimiter = " ")

        start_time_episode = time.time()
        steps = 0
        for e in range(self.nepisodes):
            current_state = self.env.reset()
            total_reward=0
            all_done = False
            done = False
            if self.leader is True: # for leaders
                done = True
            
            root = 0
            if self.leader is True: 
                root = MPI.PROC_NULL
            if rank == 0:
                root = MPI.ROOT
             
            while all_done!=True:

                new_data = [] 
                worker_data = []
                worker_state = None
              
                ### workers
                if self.leader is False: 
                    done = intracomm.allreduce(done, op=MPI.LAND)
                    if done != True:
                        action = self.agent.action(current_state)
                        next_state, reward, done, _ = self.env.step(action)
                        total_reward += reward
                        worker_state = (action, reward, next_state, done, total_reward)

                ### communicate from workers to remote leader of workers
                if self.leader is True: 
                    for i in range(ncolors-1):
                        worker_data = intercomm[i].gather(worker_state, root=root)
                else:
                    worker_data = intercomm[0].gather(worker_state, root=root)
                
                ## Learner (also a leader) ##
                if rank == 0:
                    for wdata in worker_data:
                        if wdata is not None:
                            new_data.append([current_state, wdata[0], wdata[1], wdata[2], wdata[3], wdata[4]])

                    ## Push memories to learner ##
                    if new_data is not None:
                        for data in new_data:
                            self.agent.remember(data[0],data[1],data[2],data[3],data[4])
                    ## Train learner ##
                    self.agent.train()
                    rank0_memories = len(self.agent.memory)
                    target_weights = self.agent.get_weights()
           
                
                ### communicate from remote leader to local leader of workers
                ## broadcast the memory size and the model weights to the workers  ##       
                if self.leader is True: 
                    for i in range(ncolors-1):
                        rank0_memories = intercomm[i].bcast(rank0_memories, root=root)
                else:
                    rank0_memories = intercomm[0].bcast(rank0_memories, root=root)
                
                if self.leader is True: 
                    for i in range(ncolors-1):
                        current_weights = intercomm[i].bcast(target_weights, root=root)
                else:
                    current_weights = intercomm[0].bcast(target_weights, root=root)
                
                if self.leader is True: 
                    for i in range(ncolors-1):
                        new_data = intercomm[i].bcast(new_data, root=root)
                else:
                    new_data = intercomm[0].bcast(new_data, root=root)       
 
                ## Set the model weight for all the workers
                if self.leader is False: 
                    if current_weights is not None and rank0_memories is not None and rank0_memories>30:                            
                        self.agent.set_weights(current_weights)

                    ## Update state
                    if done != True:
                        current_state = next_state
                    
                    ## Save memory for offline analysis
                    # current_state,action,reward,next_state,total_reward,done
                    if new_data is not None:
                        for data in new_data:
                            train_writer.writerow([current_state,data[1],data[2],data[3],data[5],data[4]])
                            train_file.flush()
                           
                steps += 1
                if steps >= self.nsteps:
                    done = True

                ## Exit criteria
                all_done = comm.allreduce(done, op=MPI.LAND)
            
            end_time_episode = time.time()
            mtim = end_time_episode - start_time_episode 
            ptim = comm.reduce(mtim, op=MPI.SUM, root=0)
            if rank == 0:
                logger.info('Average execution time (in secs.) for %s episodes: %s ' % (str(rank), str(e), str(ptim / size)))

            if self.leader is False:
                train_file.close()

    def run(self, run_type):
        if self.agent!=None:
            self.agent.set_agent()

        if self.env!=None:
            self.env.set_env()

        if run_type == 'static-two-groups':
            self.run_exarl_two_groups(self.intracomm, self.intercomm)
            self.intracomm.Free()
            self.intercomm.Free()
        elif run_type == 'static-multi-groups':
            self.run_exarl_multi_groups(self.intracomm, self.intercomm)
            self.intracomm.Free()
            ncolors = self.mpi_children_per_parent+1
            if self.leader is True: 
                for i in range(ncolors-1):
                    self.intercomm[i].Free()
            else:
                self.intercomm[0].Free()
        elif run_type == 'dynamic':
            self.run_exarl(self.world_comm)
        else:
            if self.agent_comm != MPI.COMM_NULL:
                self.run_exarl(self.agent_comm)
                self.agent_comm.Free()

        if self.env_comm != MPI.COMM_NULL:
            self.env_comm.Free()
