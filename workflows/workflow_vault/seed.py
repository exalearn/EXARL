import time
import csv
from mpi4py import MPI
import sys
import exarl as erl

import utils.log as log
import utils.candleDriver as cd
logger = log.setup_logger(__name__, cd.run_params['log_level'])


class SEED(erl.ExaWorkflow):
    def __init__(self):
        print('Class SEED')

    def run(self, learner):
        comm = MPI.COMM_WORLD
        sys.exit('TBD')

        filename_prefix = 'ExaLearner_' + 'Episodes%s_Steps%s_Rank%s_memory_v1' % (
            str(learner.nepisodes), str(learner.nsteps), str(comm.rank))
        train_file = open(learner.results_dir + '/' +
                          filename_prefix + ".log", 'w')
        train_writer = csv.writer(train_file, delimiter=" ")
        # print('learner.world_comm.rank:',learner.world_comm.rank)

        for e in range(learner.nepisodes):

            rank0_memories = 0
            rank0_epsilon = 0
            target_weights = None
            current_state = learner.env.reset()
            total_reward = 0
            done = False
            all_done = False

            start_time_episode = time.time()
            steps = 0
            while all_done != True:
                # All workers
                if done != True:
                    action, policy_type = learner.agent.action(current_state)
                    next_state, reward, done, _ = learner.env.step(action)
                    total_reward += reward
                    memory = (current_state, action, reward,
                              next_state, done, total_reward)

                new_data = comm.gather(memory, root=0)
                logger.info('Rank[%s] - Memory length: %s ' %
                            (str(comm.rank), len(learner.agent.memory)))

                # Learner
                if comm.rank == 0:
                    # Push memories to learner
                    for data in new_data:
                        # print(data)
                        learner.agent.remember(
                            data[0], data[1], data[2], data[3], data[4])
                        # Train learner
                        # learner.agent.train()
                        rank0_epsilon = learner.agent.epsilon
                        rank0_memories = len(learner.agent.memory)
                        target_weights = learner.agent.get_weights()
                        if rank0_memories % (comm.size) == 0:
                            learner.agent.save(
                                learner.results_dir + '/' + filename_prefix + '.h5')

                # Broadcast the memory size and the model weights to the workers
                rank0_epsilon = comm.bcast(rank0_epsilon, root=0)
                rank0_memories = comm.bcast(rank0_memories, root=0)
                current_weights = comm.bcast(target_weights, root=0)

                logger.info('Rank[%s] - rank0 memories: %s' %
                            (str(comm.rank), str(rank0_memories)))

                # Set the model weight for all the workers
                # and rank0_memories > 30:# and rank0_memories%(size)==0:
                if comm.rank > 0:
                    logger.info(
                        '## Rank[%s] - Updating weights ##' % str(comm.rank))
                    learner.agent.set_weights(current_weights)
                    learner.agent.epsilon = rank0_epsilon

                # Save memory for offline analysis
                train_writer.writerow([current_state, action, reward, next_state,
                                       total_reward, done, e, steps, policy_type, rank0_epsilon])
                train_file.flush()

                # Update state
                current_state = next_state
                logger.info('Rank[%s] - Total Reward:%s' %
                            (str(comm.rank), str(total_reward)))

                # Save Learning target model
                if comm.rank == 0:
                    learner.agent.save(learner.results_dir +
                                       '/' + filename_prefix + '.h5')

                steps += 1
                if steps >= learner.nsteps:
                    done = True

                all_done = comm.allreduce(done, op=MPI.LAND)

            end_time_episode = time.time()
            logger.info('Rank[%s] run-time for episode %s: %s ' %
                        (str(comm.rank), str(e), str(end_time_episode - start_time_episode)))

        train_file.close()
