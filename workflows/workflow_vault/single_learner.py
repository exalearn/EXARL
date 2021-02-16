import time
import csv
import exarl as erl
from exarl import ExaComm
from utils.introspect import ib

import utils.log as log
import utils.candleDriver as cd
logger = log.setup_logger(__name__, cd.run_params['log_level'])


class SYNC(erl.ExaWorkflow):
    def __init__(self):
        print('Class SYNC learner')

    def run(self, learner):
        comm = ExaComm.global_comm

        filename_prefix = 'ExaLearner_' + 'Episodes%s_Steps%s_Rank%s_memory_v1' % (
            str(learner.nepisodes), str(learner.nsteps), str(comm.rank))
        train_file = open(learner.results_dir + '/' +
                          filename_prefix + ".log", 'w')
        train_writer = csv.writer(train_file, delimiter=" ")

        ########################################
        # Set target model the sample for all #
        ########################################
        target_weights = None
        if comm.rank == 0:
            target_weights = learner.agent.get_weights()

        ####################################
        # Send and set to all other agents #
        ####################################
        current_weights = comm.bcast(target_weights, root=0)
        learner.agent.set_weights(current_weights)

        #####################
        # Variables for all #
        #####################
        rank0_epsilon  = 0

        ######################
        # Loop over episodes #
        ######################
        for e in range(learner.nepisodes):

            ################################
            # Reset variables each episode #
            ################################
            current_state = learner.env.reset()
            total_reward = 0
            steps        = 0
            done         = False
            all_done     = False

            start_time_episode = time.time()

            while all_done != True:
                # All workers
                reward = -9999
                memory = (current_state, None, reward, None, done, 0)
                if done != True:
                    action, policy_type = learner.agent.action(current_state)
                    ib.update("Sync_Env_Inference", 1)
                    ib.startTrace("step", 0)
                    next_state, reward, done, _ = learner.env.step(action)
                    ib.stopTrace()
                    ib.update("Sync_Env_Step", 1)
                    total_reward += reward
                    memory = (current_state, action, reward,
                              next_state, done, total_reward)

                batch_data = []
                if memory[2] != -9999:
                    learner.agent.remember(
                        memory[0], memory[1], memory[2], memory[3], memory[4])
                    # TODO: we need a memory class to scale
                    batch_data = next(learner.agent.generate_data())
                    ib.update("Sync_Env_Generate_Data", 1)
                    logger.info(
                        'Rank[{}] - Generated data: {}'.format(comm.rank, len(batch_data[0])))
                logger.info(
                    'Rank[{}] - Memories: {}'.format(comm.rank, len(learner.agent.memory)))

                # TODO: gatherall to share memories with all agents
                # new_data = comm.allgather(memory)
                # logger.info('Rank [{}] - allgather memories: {}'.format(comm.rank, new_data))

                # for data in new_data:
                #    # logger.info('Rank [{}] - Memories/[2]: {} / {}'.format(comm.rank,data,data[2]))
                #    if data[2] != -9999:
                #        # TODO: Improve remember function
                #        learner.agent.remember(data[0], data[1], data[2], data[3], data[4])
                # logger.info('Rank[%s] - Memory length: %s ' % (str(comm.rank),len(learner.agent.memory)))

                # print('batch_data {}'.format(batch_data))

                # TODO: gather the generated data for the learner
                # TODO: should it be an isend irecv ?
                new_batch = comm.gather(batch_data, 0)

                # Learner
                if comm.rank == 0:
                    # Push memories to learner
                    for batch in new_batch:
                        learner.agent.train(batch)
                    ib.update("Sync_Learner_Train", 1)
                    learner.agent.target_train()
                    ib.update("Sync_Learner_Target_Train", 1)
                    rank0_epsilon = learner.agent.epsilon
                    target_weights = learner.agent.get_weights()
                    # if rank0_memories%(comm.size) == 0:
                    #    learner.agent.save(learner.results_dir+'/'+filename_prefix+'.h5')

                # Broadcast the memory size and the model weights to the workers
                rank0_epsilon = comm.bcast(rank0_epsilon, root=0)
                current_weights = comm.bcast(target_weights, root=0)

                # Set the model weight for all the workers
                # if comm.rank > 0:# and rank0_memories > 30:# and rank0_memories%(size)==0:
                #    logger.info('## Rank[%s] - Updating weights ##' % str(comm.rank))
                learner.agent.set_weights(current_weights)
                learner.agent.epsilon = rank0_epsilon

                # Update state
                current_state = next_state
                logger.info('Rank[%s] - Total Reward:%s' %
                            (str(comm.rank), str(total_reward)))

                # Save Learning target model
                if comm.rank == 0:
                    learner.agent.save(learner.results_dir + '/' + filename_prefix + '.h5')

                steps += 1
                if steps >= learner.nsteps:
                    done = True

                # Save memory for offline analysis
                if reward != -9999:
                    train_writer.writerow([time.time(), current_state, action, reward,
                                           next_state, total_reward, done, e, steps, policy_type, rank0_epsilon])
                    train_file.flush()

                all_done = comm.allreduce(done)
                ib.update("Sync_Learner_Episode", 1)

            end_time_episode = time.time()
            logger.info('Rank[%s] run-time for episode %s: %s ' %
                        (str(comm.rank), str(e), str(end_time_episode - start_time_episode)))
            logger.info('Rank[%s] run-time for episode per step %s: %s '
                        % (str(comm.rank), str(e), str((end_time_episode - start_time_episode) / steps)))
        train_file.close()
