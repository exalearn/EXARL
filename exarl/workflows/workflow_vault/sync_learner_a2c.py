# This material was prepared as an account of work sponsored by an agency of the
# United States Government.  Neither the United States Government nor the United
# States Department of Energy, nor Battelle, nor any of their employees, nor any
# jurisdiction or organization that has cooperated in the development of these
# materials, makes any warranty, express or implied, or assumes any legal
# liability or responsibility for the accuracy, completeness, or usefulness or
# any information, apparatus, product, software, or process disclosed, or
# represents that its use would not infringe privately owned rights. Reference
# herein to any specific commercial product, process, or service by trade name,
# trademark, manufacturer, or otherwise does not necessarily constitute or imply
# its endorsement, recommendation, or favoring by the United States Government
# or any agency thereof, or Battelle Memorial Institute. The views and opinions
# of authors expressed herein do not necessarily state or reflect those of the
# United States Government or any agency thereof.
#                 PACIFIC NORTHWEST NATIONAL LABORATORY
#                            operated by
#                             BATTELLE
#                             for the
#                   UNITED STATES DEPARTMENT OF ENERGY
#                    under Contract DE-AC05-76RL01830
import time
import csv
from mpi4py import MPI
import exarl as erl
import exarl.mpi_settings as mpi_settings
import exarl.utils.log as log
import exarl.utils.candleDriver as cd
from exarl.utils.profile import *
logger = log.setup_logger(__name__, cd.run_params['log_level'])


class SYNC2(erl.ExaWorkflow):
    def __init__(self):
        print('Class SYNC learner')

    @PROFILE
    def run(self, workflow):
        comm = MPI.COMM_WORLD

        filename_prefix = 'ExaLearner_' + 'Episodes%s_Steps%s_Rank%s_memory_v1' % (
            str(workflow.nepisodes), str(workflow.nsteps), str(comm.rank))
        train_file = open(workflow.results_dir + '/' +
                          filename_prefix + ".log", 'w')
        train_writer = csv.writer(train_file, delimiter=" ")

        # Set target model the sample for all
        target_weights = None

        if mpi_settings.is_learner():
            workflow.agent.set_learner()
            target_weights = workflow.agent.get_weights()

        # Send and set to all other agents
        current_weights = comm.bcast(target_weights, root=0)
        workflow.agent.set_weights(current_weights)

        # Variables for all
        rank0_epsilon = 0

        # Loop over episodes
        for e in range(workflow.nepisodes):

            # Reset variables each episode
            current_state = workflow.env.reset()
            total_reward  = 0
            steps         = 0
            done          = False
            all_done      = False

            start_time_episode = time.time()

            while all_done != True:
                # All workers
                reward = -9999
                memory = (current_state, None, reward, None, done, 0)
                if done != True:
                    action, policy_type = workflow.agent.action(current_state)
                    next_state, reward, done, _ = workflow.env.step(action)
                    total_reward += reward
                    memory = (current_state, action, reward,
                              next_state, done, total_reward)

                batch_data = []
                if memory[2] != -9999:
                    workflow.agent.remember(
                        memory[0], memory[1], memory[2], memory[3], memory[4])

                # Update state
                current_state = next_state
                logger.info('Rank[%s] - Total Reward:%s' %
                            (str(comm.rank), str(total_reward)))

                steps += 1
                if steps >= workflow.nsteps:
                    done = True

                # Save memory for offline analysis
                if reward != -9999:
                    train_writer.writerow([time.time(), current_state, action, reward,
                                           next_state, total_reward, done, e, steps, policy_type, rank0_epsilon])
                    train_file.flush()

                all_done = comm.allreduce(done, op=MPI.LAND)

            batch_data = [workflow.agent.state_memory, workflow.agent.action_memory, workflow.agent.reward_memory]
            workflow.agent.reset_lists()
            new_batch = comm.gather(batch_data, root=0)

            # Learner
            if mpi_settings.is_learner():
                # Push memories to learner
                for batch in new_batch:
                    train_return = workflow.agent.train(batch)

                if train_return is not None:
                    # indices, loss = train_return
                    workflow.agent.set_priorities(*train_return)
                workflow.agent.target_train()
                rank0_epsilon = workflow.agent.epsilon
                target_weights = workflow.agent.get_weights()
                # if rank0_memories%(comm.size) == 0:
                #    workflow.agent.save(workflow.results_dir+'/'+filename_prefix+'.h5')

            # Broadcast the memory size and the model weights to the workers
            rank0_epsilon = comm.bcast(rank0_epsilon, root=0)
            current_weights = comm.bcast(target_weights, root=0)

            # Set the model weight for all the workers
            workflow.agent.set_weights(current_weights)
            workflow.agent.epsilon = rank0_epsilon

            # Save Learning target model
            if mpi_settings.is_learner():
                workflow.agent.save(workflow.results_dir + '/' + filename_prefix + '.h5')

            end_time_episode = time.time()
            logger.info('Rank[%s] run-time for episode %s: %s ' %
                        (str(comm.rank), str(e), str(end_time_episode - start_time_episode)))
            logger.info('Rank[%s] run-time for episode per step %s: %s '
                        % (str(comm.rank), str(e), str((end_time_episode - start_time_episode) / steps)))
        train_file.close()
