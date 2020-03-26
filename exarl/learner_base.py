# © (or copyright) 2020. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
# National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
# Department of Energy/National Nuclear Security Administration. All rights in the program are
# reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
# Security Administration. The Government is granted for itself and others acting on its behalf a
# nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
# derivative works, distribute copies to the public, perform publicly and display publicly, and 
# to permit others to do so.


import gym
import exarl as erl
import time
import os
import csv

from mpi4py import MPI
import json

class ExaLearner():

    def __init__(self, agent_id, env_id, cfg='envs/env_vault/env_cfg/env_setup.json'):
        ## Default training 
        self.nepisodes=1
        self.nsteps=10
        self.results_dir='./results/'
        self.do_render=False
        
        ## Setup agent and environents
        self.agent_id = agent_id
        self.env_id   = env_id
        self.agent, self.env = erl.make(agent_id, env_id)
        self.env._max_episode_steps=self.nsteps

        self.cfg = cfg
        with open(self.cfg) as json_file:
            data = json.load(json_file)
        self.mpi_children_per_parent = int(data['mpi_children_per_parent']) \
                                       if 'mpi_children_per_parent' in data.keys() \
                                       else 0
        
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
        self.agent.set_results_dir(self.results_dir)
        ## Set for env
        self.env.set_results_dir(self.results_dir)

    def render_env(self):
        self.do_render=True

    def run_static(self, train_file, train_writer, comm=MPI.COMM_WORLD):
        rank0_memories = 0
        target_weights = None

        ttim = 0.0
        stim = time.time()

        for e in range(self.nepisodes):
            current_state = self.env.reset()
            total_reward=0
            done = False
            all_done = False

            while all_done!=True:
            
                if done != True:
                    action = self.agent.action(current_state)
                    next_state, reward, done, _ = self.env.step(action)
                
                total_reward+=reward
                memory = (current_state, action, reward, next_state, done, total_reward)
                new_data = comm.gather(memory, root=0)
            
                ## Learner ##
                if comm.rank==0:
                
                    lstim = time.time()
                    ## Push memories to learner ##
                    for data in new_data:
                        self.agent.remember(data[0],data[1],data[2],data[3],data[4])
                    
                        ## Train learner ##
                        self.agent.train()
                        rank0_memories = len(self.agent.memory)
                        target_weights = self.agent.get_weights()
                        letim = time.time()
                        ttim += (letim - lstim)
           
                comm.barrier()

                ## Broadcast the memory size and the model weights to the workers  ##
                rank0_memories = comm.bcast(rank0_memories, root=0)
                current_weights = comm.bcast(target_weights, root=0)
                
                ## Set the model weight for all the workers
                if comm.rank>0 and rank0_memories>30:                            
                    self.agent.set_weights(current_weights)

                ## Update state
                if done != True:
                    current_state = next_state
                    
                ## Save memory for offline analysis
                train_writer.writerow([current_state,action,reward,next_state,total_reward,done])
                train_file.flush()
           
                ## Exit criteria
                all_done = comm.allreduce(done, op=MPI.LAND)
    
        etim = time.time()
        mtim  = etim - stim
        ptim = comm.reduce(mtim, op=MPI.SUM, root=0)

        if comm.rank==0:
	    # create a file to load perf details
            perf_file_name = 'ExaLearner_' + 'Episode%s_Steps%s_Size%s_memory_v1_perf' % (str(self.nepisodes), str(self.nsteps), str(comm.size))
            print('Average time taken for %s episodes across %s ranks: %s secs' % (self.nepisodes, size, float(ptim / size)), file=open(perf_file_name + ".txt", 'w+'))
            print('Accumulated training time for the single learner over all the episodes on process 0: %s secs' % (ttim), file=open(perf_file_name + ".txt", 'a'))
              

    ### Untested, future purge candidate, at present 
    ### calls the static code above 
    def run_dynamic(self, train_file, train_writer, comm=MPI.COMM_WORLD):
         self.run_static(train_file, train_writer, comm)

    # top-level run
    def run(self, type):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        filename_prefix = 'ExaLearner_' + 'Episode%s_Steps%s_Rank%s_memory_v1_%s' % (str(self.nepisodes), str(self.nsteps), str(rank), str(type))
        train_file = open(self.results_dir+'/'+filename_prefix + ".log", 'w')
        train_writer = csv.writer(train_file, delimiter = " ")
        color = MPI.UNDEFINED

        if type == 'static':
            ### Create new comm of leaders, make sure rank 0 (rank that controls learner) is included
            if rank == 0 or rank%(self.mpi_children_per_parent+1) == 0:
                color = size
                new_comm = comm.Split(color, rank)
            comm.barrier()
            self.run_static(train_file, train_writer, new_comm)
            if new_comm != MPI.COMM_NULL:
                new_comm.Free()
        elif type == 'dynamic':
            self.run_dynamic(train_file, train_writer)

        comm.barrier()
            
        ## Save Learning target model
        if comm.rank==0:
            self.agent.save(self.results_dir+filename_prefix+'.h5')
            train_file.close()
