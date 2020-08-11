import time
import csv
from mpi4py import MPI
import numpy as np
import logging
import sys
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RL-Logger')
logger.setLevel(logging.INFO)

def run_async_learner(self, comm):

        # Set target model the sample for all
        target_weights = None
        if comm.rank == 0:
            target_weights = self.agent.get_weights()
        
        # Send and set to all other agents
        current_weights = comm.bcast(target_weights, root=0)
        self.agent.set_weights(current_weights)

        # Variables for all
        episode = 0
        rank0_memories = 0
        rank0_epsilon  = 0

        ## Round-Robin Scheduler
        if comm.rank == 0:
                total_learner_time = 0
                
                while episode < self.nepisodes:
                        start_learner_time = time.time()
                        print("Running scheduler/learner episode: {}".format(episode))
                        
                        # Receive the rank of the worker ready for more work
                        recv_data = comm.recv(source=MPI.ANY_SOURCE, tag=1)
                        whofrom = recv_data[0]
                        step = recv_data[1]
                        #print('whofrom: {}'.format(whofrom))
                        # Send target weights

                        rank0_epsilon = self.agent.epsilon
                        target_weights = self.agent.get_weights()
                        comm.send([episode, rank0_epsilon, target_weights], dest = whofrom)
                        
                        # Receive batch
                        recv_data = comm.recv(source=MPI.ANY_SOURCE, tag=2)
                        batch = recv_data[0]
                        done = recv_data[1]
                        
                        # Train
                        self.agent.train(batch)
                        self.agent.target_train()
                        
                        # Increment episode when complete
                        #if done:
                        if step==0:
                                episode += 1
                                
                        stop_learner_time = time.time()
                        total_learner_time += stop_learner_time - start_learner_time
                        
                print("Finishing up ...")
                episode = -1
                for s in range(1, comm.Get_size()):
                        comm.send([episode, 0], dest=s)
                
                print('Learner time: {}'.format(total_learner_time))

        else:                   
                # Setup logger
                filename_prefix = 'ExaLearner_' + 'Episodes%s_Steps%s_Rank%s_memory_v1' \
                                  % ( str(self.nepisodes), str(self.nsteps), str(comm.rank))
                train_file = open(self.results_dir+'/'+filename_prefix + ".log", 'w')
                train_writer = csv.writer(train_file, delimiter = " ")
                
                while episode != -1:
                        # Reset variables each episode
                        self.env.seed(0)
                        current_state   = self.env.reset()
                        total_reward    = 0
                        steps           = 0
                        done            = False

                        # Steps in an episode
                        while steps < self.nsteps:                                
                                # Indicate availability by sending rank to the learner
                                comm.send([comm.rank, steps], dest=0, tag=1)

                                # Receive target weights
                                recv_data = comm.recv(source=0)
                                if steps==0:
                                        episode = recv_data[0]
                                        
                                if recv_data[0] == -1:
                                        episode = -1
                                        break
                                
                                self.agent.epsilon = recv_data[1]
                                self.agent.set_weights(recv_data[2])
                                
                                action, policy_type = 0, -11 #self.agent.action(current_state)
                                next_state, reward, done, _ = self.env.step(action)
                                total_reward += reward
                                memory = (current_state, action, reward, next_state, done, total_reward)
                                
                                #batch_data = []
                                self.agent.remember(memory[0], memory[1], memory[2], memory[3], memory[4])
                    
                                batch_data = next(self.agent.generate_data())
                                logger.info('Rank[{}] - Generated data: {}'.format(comm.rank, len(batch_data[0])))
                                logger.info('Rank[{}] - Memories: {}'.format(comm.rank,len(self.agent.memory)))

                                # Send batched memories
                                comm.send([batch_data, done], dest=0, tag=2)

                                # Update state
                                current_state = next_state
                                steps += 1
                                if steps >= self.nsteps:
                                        done = True

                                logger.info('Rank[%s] - Total Reward:%s' % (str(comm.rank),str(total_reward)))
                                logger.info('Rank[%s] - Episode/Step:%s/%s' % (str(comm.rank),str(episode),str(steps)))

                                train_writer.writerow([time.time(),current_state,action,reward,next_state,total_reward, \
                                                       done, steps, policy_type, rank0_epsilon])
                                train_file.flush()

                                # Break for loop if done
                                if done:
                                        break


                train_file.close()
                                

                                
