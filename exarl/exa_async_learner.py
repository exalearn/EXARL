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
        rank0_memories = 0
        rank0_epsilon  = 0

        ## Round-Robin Scheduler
        if rank == 0:
                while 1:
                        if episode < self.nepisodes:
                                print("Running scheduler/learner")
                                done = False
                                while done != True: 
                                        # Receive batch
                                        req = comm.Irecv(recv_data, source=MPI.ANY_SOURCE)
                                        req.Wait()
                                        whofrom = recv_data[0]
                                        batch = recv_data[1]
                                        done = recv_data[2]
                                
                                        # Train
                                        self.agent.train(batch)
                                        self.agent.target_train()
                                        rank0_epsilon = self.target.epsilon
                                        target_weights =self.agent.get_weights()
                                        
                                        # Send target weights
                                        comm.Isend([rank0_epsilon, target_weights], source = whofrom)
                                
                                # Increment episode
                                episode += 1
                        else:
                                print("Finishing up ...")
                                episode[0] = -1
                                for s in range(1, comm.Get_size()):
                                        comm.Isend(episode, dest=s)
                                break

        else:                   
                # Setup logger
                filename_prefix = 'ExaLearner_' + 'Episodes%s_Steps%s_Rank%s_memory_v1' 
                                  % ( str(self.nepisodes), str(self.nsteps), str(comm.rank))
                train_file = open(self.results_dir+'/'+filename_prefix + ".log", 'w')
                train_writer = csv.writer(train_file, delimiter = " ")
                
                # Reset variables each episode
                current_state = self.env.reset()
                total_reward = 0
                steps        = 0
                done         = False
                all_done     = False

                while episode[0] != -1:

                        # Steps in an episode
                        for i in range(self.nsteps):
                                action = self.agent.action(current_state)
                                next_state, reward, done, _ = self.env.step(action)
                                total_reward+=reward
                                memory = (current_state, action, reward, next_state, done, total_reward)
                                
                                batch_data = []
                                self.agent.remember(memory[0], memory[1], memory[2], memory[3], memory[4])
                    
                                batch_data = next(self.agent.generate_data())
                                logger.info('Rank[{}] - Generated data: {}'.format(comm.rank, len(batch_data[0])))
                                logger.info('Rank[{}] - Memories: {}'.format(comm.rank,len(self.agent.memory)))

                                # Send batched memories
                                req = comm.Isend([comm.rank, batch_data, done], dest=0)
                                
                                # Receive target weights
                                req = comm.Irecv(recv_data, source=0)
                                req.Wait()

                                self.agent.epsilon = recv_data[0]
                                self.agent.set_weights(recv_data[1])
                                
                                # Update state
                                current_state = next_state
                                logger.info('Rank[%s] - Total Reward:%s' % (str(comm.rank),str(total_reward)))

                                train_writer.writerow([time.time(),current_state,action,reward,next_state,total_reward,
                                                       done, e, steps, policy_type, rank0_epsilon])
                                train_file.flush()

                                # Break loop if done
                                if done:
                                        break
                        
                        logger.info('Rank[%s] run-time for episode %s: %s ' 
                                    % (str(comm.rank), str(e), str(end_time_episode - start_time_episode)))
                        logger.info('Rank[%s] run-time for episode per step %s: %s '
                                    % (str(comm.rank), str(e), str((end_time_episode - start_time_episode)/steps)))

                train_file.close()
                                

                                
