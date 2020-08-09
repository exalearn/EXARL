import time
import csv
from mpi4py import MPI
import numpy as np
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RL-Logger')
logger.setLevel(logging.INFO)

def run_async_learner(self, comm):

        ## Initial setup

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
                while 1:
                        if episode < self.nepisodes:
                                print("Running scheduler/learner")
                                done = False
                                while done != True: 
                                        # Receive batch
                                        recv_data = comm.recv(source=MPI.ANY_SOURCE)
                                        whofrom = recv_data[0]
                                        batch = recv_data[1]
                                        done = recv_data[2]
                                
                                        # Train
                                        self.agent.train(batch)
                                        self.agent.target_train()
                                        rank0_epsilon = self.agent.epsilon
                                        target_weights =self.agent.get_weights()
                                        
                                        # Send target weights
                                        comm.send([episode, rank0_epsilon, target_weights], dest = whofrom)
                                
                                # Increment episode
                                episode += 1
                        else:
                                print("Finishing up ...")
                                episode = -1
                                for s in range(1, comm.Get_size()):
                                        comm.send([episode, 0], dest=s)
                                break

        else:                   
                # Setup logger
                filename_prefix = 'ExaLearner_' + 'Episodes%s_Steps%s_Rank%s_memory_v1' \
                                  % ( str(self.nepisodes), str(self.nsteps), str(comm.rank))
                train_file = open(self.results_dir+'/'+filename_prefix + ".log", 'w')
                train_writer = csv.writer(train_file, delimiter = " ")
                
                while episode != -1:
                        # Reset variables each episode
                        current_state = self.env.reset()
                        total_reward = 0
                        steps = 0
                        done         = False

                        # Steps in an episode
                        for i in range(self.nsteps):
                                action, policy_type = self.agent.action(current_state)
                                next_state, reward, done, _ = self.env.step(action)
                                total_reward += reward
                                memory = (current_state, action, reward, next_state, done, total_reward)
                                
                                batch_data = []
                                self.agent.remember(memory[0], memory[1], memory[2], memory[3], memory[4])
                    
                                batch_data = next(self.agent.generate_data())
                                logger.info('Rank[{}] - Generated data: {}'.format(comm.rank, len(batch_data[0])))
                                logger.info('Rank[{}] - Memories: {}'.format(comm.rank,len(self.agent.memory)))

                                # Send batched memories
                                comm.send([comm.rank, batch_data, done], dest=0)
                                
                                # Receive target weights
                                recv_data = comm.recv(source=0)
                                episode = recv_data[0]
                                if episode == -1:
                                        break

                                self.agent.epsilon = recv_data[1]
                                self.agent.set_weights(recv_data[2])
                                
                                # Update state
                                current_state = next_state
                                steps += 1

                                logger.info('Rank[%s] - Total Reward:%s' % (str(comm.rank),str(total_reward)))

                                train_writer.writerow([time.time(),current_state,action,reward,next_state,total_reward, \
                                                       done, steps, policy_type, rank0_epsilon])
                                train_file.flush()

                                # Break for loop if done
                                if done:
                                        break


                train_file.close()
                                

                                
