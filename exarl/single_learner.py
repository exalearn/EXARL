import time
import csv
from mpi4py import MPI

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RL-Logger')
logger.setLevel(logging.INFO)

import multiprocessing

def run_single_learner(self):
        comm = MPI.COMM_WORLD

        filename_prefix = 'ExaLearner_' + 'Episodes%s_Steps%s_Rank%s_memory_v1' % ( str(self.nepisodes), str(self.nsteps), str(comm.rank))
        train_file = open(self.results_dir+'/'+filename_prefix + ".log", 'w')
        train_writer = csv.writer(train_file, delimiter = " ")

        #########################################
        ## Set target model the sample for all ##
        #########################################
        target_weights = None
        if comm.rank == 0:
            target_weights = self.agent.get_weights()

        ######################################
        ## Send and set to all other agents ##
        ######################################
        current_weights = comm.bcast(target_weights, root=0)
        self.agent.set_weights(current_weights)

        #######################
        ## Variables for all ##
        #######################
        rank0_memories = 0
        rank0_epsilon  = 0

        ########################
        ## Loop over episodes ##
        ########################
        for e in range(self.nepisodes):

            ##################################
            ## Reset variables each episode ##
            ##################################
            current_state = self.env.reset()
            total_reward = 0
            steps        = 0
            done         = False
            all_done     = False

            start_time_episode = time.time()

            while all_done != True:
                ## All workers ##
                reward = -9999
                memory = (current_state, None, reward, None, done, 0)
                if done != True:
                    action, policy_type = self.agent.action(current_state)
                    next_state, reward, done, _ = self.env.step(action)
                    total_reward += reward
                    memory = (current_state, action, reward, next_state, done, total_reward)

                ##
                batch_data = []
                if memory[2] != -9999:
                    self.agent.remember(memory[0], memory[1], memory[2], memory[3], memory[4])
                    ## TODO: we need a memory class to scale
                    batch_data = next(self.agent.generate_data())
                    logger.info('Rank[{}] - Generated data: {}'.format(comm.rank, len(batch_data[0])))
                logger.info('Rank[{}] - Memories: {}'.format(comm.rank,len(self.agent.memory)))

                ## TODO: gatherall to share memories with all agents 
                #new_data = comm.allgather(memory)
                #logger.info('Rank [{}] - allgather memories: {}'.format(comm.rank, new_data))

                #for data in new_data:
                #    #logger.info('Rank [{}] - Memories/[2]: {} / {}'.format(comm.rank,data,data[2]))
                #    if data[2] != -9999:
                #        ## TODO: Improve remember function
                #        self.agent.remember(data[0], data[1], data[2], data[3], data[4])
                #logger.info('Rank[%s] - Memory length: %s ' % (str(comm.rank),len(self.agent.memory)))

                #print('batch_data {}'.format(batch_data))

                ## TODO: gather the generated data for the learner
                ## TODO: should it be an isend irecv ?
                new_batch = comm.gather(batch_data,root=0)

                ## Learner ##
                if comm.rank == 0:
                    ## Push memories to learner ##
                    for batch in new_batch:
                        self.agent.train(batch)
                    self.agent.target_train()
                    rank0_epsilon = self.agent.epsilon
                    target_weights = self.agent.get_weights()
                        #if rank0_memories%(comm.size) == 0:
                        #    self.agent.save(self.results_dir+'/'+filename_prefix+'.h5')

                ## Broadcast the memory size and the model weights to the workers  ##
                rank0_epsilon = comm.bcast(rank0_epsilon, root=0)
                current_weights = comm.bcast(target_weights, root=0)

                ## Set the model weight for all the workers
                #if comm.rank > 0:# and rank0_memories > 30:# and rank0_memories%(size)==0:
                #    logger.info('## Rank[%s] - Updating weights ##' % str(comm.rank))
                self.agent.set_weights(current_weights)
                self.agent.epsilon = rank0_epsilon

                ## Update state
                current_state = next_state
                logger.info('Rank[%s] - Total Reward:%s' % (str(comm.rank),str(total_reward)))
                
                ## Save Learning target model
                if comm.rank == 0:
                    self.agent.save(self.results_dir+'/'+filename_prefix+'.h5')

                steps += 1
                if steps >= self.nsteps:
                    done = True

                ## Save memory for offline analysis
                if reward!=-9999:
                    train_writer.writerow([time.time(),current_state,action,reward,next_state,total_reward, done, e, steps, policy_type, rank0_epsilon])
                    train_file.flush()

                all_done = comm.allreduce(done, op=MPI.LAND)

            end_time_episode = time.time()
            logger.info('Rank[%s] run-time for episode %s: %s ' % (str(comm.rank), str(e), str(end_time_episode - start_time_episode)))
            logger.info('Rank[%s] run-time for episode per step %s: %s '
                        % (str(comm.rank), str(e), str((end_time_episode - start_time_episode)/steps)))
        train_file.close()
