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
                                       else 1
        self.omp_num_threads = int(data['omp_num_threads']) \
                                       if 'omp_num_threads' in data.keys() \
                                       else 1       
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
    
    def run_static_omp(self, train_file, train_writer, comm=MPI.COMM_WORLD):
        rank0_memories = 0
        target_weights = None
        ttim = 0.0
        
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
        return ttim

    def run_static(self, train_file, train_writer, worker_begin, intracomm, intercomm):
        rank0_memories = 0
        target_weights = None
        ttim = 0.0
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        for e in range(self.nepisodes):
            current_state = self.env.reset()
            total_reward=0
            done = True # for leaders
            if rank >= worker_begin:
                done = False
            all_done = False
            root = MPI.ROOT

            while all_done!=True:

                worker_state = None
                new_data = [] 

                ### workers
                if rank >= worker_begin:
                    done = intracomm.allreduce(done, op=MPI.LAND)
                    if done != True:
                        action = self.agent.action(current_state)
                        next_state, reward, done, _ = self.env.step(action, intracomm)
                        total_reward += reward
                        worker_state = (action, reward, next_state, done, total_reward)

                ### communicate from workers to remote leader of workers
                root = 0
                if rank == 0:
                    root = MPI.ROOT # remote leader
                if rank != 0 and rank < worker_begin:
                    root = MPI.PROC_NULL
                    
                worker_state = intercomm.gather(worker_state, root=root)
                
                if rank < worker_begin: ### leaders
                    ### spread data from 0 (leader) to all in leader communicator
                    worker_data = intracomm.bcast(worker_state, root=0)
                    for wdata in worker_data:
                        if wdata is not None:
                            new_data.append([current_state, wdata[0], wdata[1], wdata[2], wdata[3], wdata[4]])
            
                ## Learner (also a leader) ##
                if comm.rank==0:
                
                    lstim = time.time()
                    ## Push memories to learner ##
                    if new_data is not None:
                        for data in new_data:
                            self.agent.remember(data[0],data[1],data[2],data[3],data[4])
                    
                        ## Train learner ##
                        self.agent.train()
                        rank0_memories = len(self.agent.memory)
                        target_weights = self.agent.get_weights()
                        letim = time.time()
                        ttim += (letim - lstim)
           
                comm.barrier()
                
                ### communicate from remote leader to local leader of workers
                ## Broadcast the memory size and the model weights to the workers  ##        
                rank0_memories = intercomm.bcast(rank0_memories, root=root)
                current_weights = intercomm.bcast(target_weights, root=root)
                new_data = intercomm.bcast(new_data, root=root)
                
                ## Set the model weight for all the workers
                if rank >= worker_begin:
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
           
                ## Exit criteria
                all_done = comm.allreduce(done, op=MPI.LAND)

        return ttim
                 
    ### Uses multiple intercomms for communicating agent comm with environment comms
    def run_static_multi_groups(self, train_file, train_writer, color, ncolors, intracomm, intercomm):
        rank0_memories = 0
        target_weights = None
        ttim = 0.0
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        root = 0
        if color == 0:
            root = MPI.PROC_NULL
        if rank == 0:
            root = MPI.ROOT

        for e in range(self.nepisodes):
            current_state = self.env.reset()
            total_reward=0
            done = True # for leaders
            if color > 0:
                done = False
            all_done = False

            while all_done!=True:

                worker_state = None
                new_data = [] 

                ### workers
                if color > 0:
                    done = intracomm.allreduce(done, op=MPI.LAND)
                    if done != True:
                        action = self.agent.action(current_state)
                        next_state, reward, done, _ = self.env.step(action, intracomm)
                        total_reward += reward
                        worker_state = (action, reward, next_state, done, total_reward)

                ### communicate from workers to remote leader of workers
                if color == 0:
                    for i in range(ncolors-1):
                        worker_state = intercomm[i].gather(worker_state, root=root)
                else:
                    worker_state = intercomm[0].gather(worker_state, root=root)
                
                if color == 0: ### leaders
                    ### spread data from 0 (leader) to all in leader communicator
                    worker_data = intracomm.bcast(worker_state, root=0)
                    for wdata in worker_data:
                        if wdata is not None:
                            new_data.append([current_state, wdata[0], wdata[1], wdata[2], wdata[3], wdata[4]])
            
                ## Learner (also a leader) ##
                if comm.rank==0:
                
                    lstim = time.time()
                    ## Push memories to learner ##
                    if new_data is not None:
                        for data in new_data:
                            self.agent.remember(data[0],data[1],data[2],data[3],data[4])
                    
                        ## Train learner ##
                        self.agent.train()
                        rank0_memories = len(self.agent.memory)
                        target_weights = self.agent.get_weights()
                        letim = time.time()
                        ttim += (letim - lstim)
           
                comm.barrier()
                
                ### communicate from remote leader to local leader of workers
                ## broadcast the memory size and the model weights to the workers  ##       
                if color == 0:
                    ## broadcast from learner root to rest of the processes in leader group
                    rank0_memories = intracomm.bcast(rank0_memories, root=0)
                    current_weights = intracomm.bcast(target_weights, root=0)
                    new_data = intracomm.bcast(new_data, root=0)
                    for i in range(ncolors-1):
                        rank0_memories = intercomm[i].bcast(rank0_memories, root=root)
                        current_weights = intercomm[i].bcast(target_weights, root=root)
                        new_data = intercomm[i].bcast(new_data, root=root)
                else:
                    rank0_memories = intercomm[0].bcast(rank0_memories, root=root)
                    current_weights = intercomm[0].bcast(target_weights, root=root)
                    new_data = intercomm[0].bcast(new_data, root=root)       
       
                ## Set the model weight for all the workers
                if color > 0:
                    ## broadcast memories/weights from worker leader to rest of the workers
                    rank0_memories = intracomm.bcast(rank0_memories, root=0)
                    current_weights = intracomm.bcast(target_weights, root=0)
                    new_data = intracomm.bcast(new_data, root=0)       

                    if rank0_memories is not None and current_weights is not None and rank0_memories>30:                            
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
           
                ## Exit criteria
                all_done = comm.allreduce(done, op=MPI.LAND)

        return ttim

    ### Untested, future purge candidate, at present 
    ### calls the static code above 
    def run_dynamic(self, train_file, train_writer, comm=MPI.COMM_WORLD):
        os.environ['OMP_NUM_THREADS']='{:d}'.format(self.omp_num_threads)
        return self.run_static_omp(train_file, train_writer, comm)        

    # top-level run
    def run(self, type):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        train_file = None
        train_writer = None
        worker_rank = MPI.PROC_NULL
        nleaders = -1

        ttim = 0.0
        stim = time.time()

        if type == 'static-two-groups':
            ### leaders: {0:worker_begin-1}, workers: {worker_begin, size-1}
            worker_begin = int(size / self.mpi_children_per_parent)
            if worker_begin == 0:
                print('[Aborting] Worker and Leader cannot have the same rank. Increase #processes and try again.')
                comm.Abort()

            if rank >= worker_begin: # only workers will update
                worker_rank = rank
                filename_prefix = 'ExaLearner_' + 'Episode%s_Steps%s_Rank%s_memory_v1_%s' % (str(self.nepisodes), str(self.nsteps), str(rank), str(type))
                train_file = open(self.results_dir+'/'+filename_prefix + ".log", 'w')
                train_writer = csv.writer(train_file, delimiter = " ")
 
            ### Identify leader colors (assumes 0 is *always* a leader)
            color = 2
            if rank >= worker_begin:
                color = 1
            ### leaders(2) and workers(1) intracomm
            intracomm = comm.Split(color, rank)
            intercomm = MPI.COMM_NULL
            nleaders = intracomm.Get_size()
            # group 1 (worker) communicates with group 2 (leader)
            if color == 1:
                intercomm = MPI.Intracomm.Create_intercomm(intracomm, 0, comm, 0)
            # group 2 (leader) communicates with group 1 (worker)
            if color == 2:
                intercomm = MPI.Intracomm.Create_intercomm(intracomm, 0, comm, worker_begin)
           
            comm.barrier()
            ttim = self.run_static(train_file, train_writer, worker_begin, intracomm, intercomm)
            intercomm.Free()
            intracomm.Free()
        elif type == 'static-multi-groups':
            ### Assumes 0 is *always* the leader of agents
            ncolors = self.mpi_children_per_parent+1
            color = int(rank % ncolors)
            if color > 0: # only workers will update
                worker_rank = rank
                filename_prefix = 'ExaLearner_' + 'Episode%s_Steps%s_Rank%s_memory_v1_%s' % (str(self.nepisodes), str(self.nsteps), str(rank), str(type))
                train_file = open(self.results_dir+'/'+filename_prefix + ".log", 'w')
                train_writer = csv.writer(train_file, delimiter = " ")
            # one-to-many group communication
            intracomm = comm.Split(color, rank)
            intercomm = [MPI.COMM_NULL]*(ncolors-1)
            nleaders = intracomm.Get_size()
            if color == 0:
                for i in range(ncolors-1):
                    intercomm[i] = MPI.Intracomm.Create_intercomm(intracomm, 0, comm, i+1)
            else:
                intercomm[0] = MPI.Intracomm.Create_intercomm(intracomm, 0, comm, 0)

            comm.barrier()
            ttim = self.run_static_multi_groups(train_file, train_writer, color, ncolors, intracomm, intercomm)
            if color == 0:
                for i in range(ncolors-1):
                    intercomm[i].Free()
            else:
                intercomm[0].Free()
            intracomm.Free()
        elif type == 'static-omp':
            ### TODO ensure that this does not override if set in job script
            os.environ['OMP_NUM_THREADS']='{:d}'.format(self.omp_num_threads)
            ttim = self.run_static_omp(train_file, train_writer, comm)
        elif type == 'dynamic': ### TODO purge candidate rm
            ttim = self.run_dynamic(train_file, train_writer)

        comm.barrier()
            
        etim = time.time()
        mtim  = etim - stim
        ptim = comm.reduce(mtim, op=MPI.SUM, root=0)

	# create a file to load perf details
        if comm.rank==0:
            perf_file_name = 'ExaLearner_' + 'Episode%s_Steps%s_Size%s_memory_v1_perf' % (str(self.nepisodes), str(self.nsteps), str(comm.size))
            print('Average time taken for %s episodes across %s ranks: %s secs' % (self.nepisodes, size, float(ptim / size)), file=open(self.results_dir + '/' +perf_file_name + ".txt", 'w+'))
            print('Accumulated training time for the single learner over all the episodes on process 0: %s secs' % (ttim), file=open(self.results_dir + '/' + perf_file_name + ".txt", 'a'))
            
            if type == 'static-two-groups' or type == 'static-multi-groups':
                print('Number of leader processes: %s, number of worker processes: %s' % (str(nleaders), str(size - nleaders)), file=open(self.results_dir + '/' + perf_file_name + ".txt", 'a'))
            if type == 'static-omp':    
                print('Number of OpenMP threads set: %s' % str(self.omp_num_threads), file=open(self.results_dir + '/' + perf_file_name + ".txt", 'a'))
 
        # save h5 file and close log files/process
        if comm.rank==0:
            h5_filename_prefix = 'ExaLearner_' + 'Episode%s_Steps%s_Size%s_memory_v1_%s' % (str(self.nepisodes), str(self.nsteps), str(size), str(type))
            self.agent.save(self.results_dir+'/'+h5_filename_prefix+'.h5')
        
        if type == 'static-two-groups' or type == 'static-multi-groups': 
            if rank == worker_rank:
                train_file.close()
        else:
            train_file.close()
