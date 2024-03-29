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
import exarl as erl
from exarl.base.comm_base import ExaComm
from exarl.utils import log
import exarl.utils.candleDriver as cd
from exarl.utils.profile import *
logger = log.setup_logger(__name__, cd.lookup_params('log_level', [3, 3]))


class SYNC(erl.ExaWorkflow):
    """Synchronous workflow class: inherits from the ExaWorkflow base class.
    It features a single learner and multiple actors. The MPI processes are statically
    launched and are split into multiple groups. The environment processes can be set
    during launchtime as a candle parameter and runs multiple multi-process environments.
    The experiences generated by the environments are gathered and sent to learner for
    training.
    """

    def __init__(self):
        print('Class SYNC learner')

    @PROFILE
    def run(self, workflow):
        """This function implements the synchronous workflow in EXARL and uses MPI
        collective communication.

        Args:
            workflow (ExaLearner type object): The ExaLearner object is used to access
            different members of the base class.

        Returns:
            None
        """
        env_comm = ExaComm.env_comm

        if ExaComm.env_comm.rank == 0:
            filename_prefix = 'ExaLearner_' + 'Episodes%s_Steps%s_Rank%s_memory_v1' % (
                str(workflow.nepisodes), str(workflow.nsteps), str(env_comm.rank))
            train_file = open(workflow.results_dir + '/' +
                              filename_prefix + ".log", 'w')
            train_writer = csv.writer(train_file, delimiter=" ")

        # Variables for all
        rank0_epsilon = 0

        # Loop over episodes
        for e in range(workflow.nepisodes):
            # Reset variables each episode
            current_state = workflow.env.reset()
            total_reward  = 0
            steps         = 0
            done          = False

            start_time_episode = time.time()

            while done != True:
                # All env ranks
                action = None
                if ExaComm.env_comm.rank == 0:
                    action, policy_type = workflow.agent.action(current_state)

                # Broadcast episode count to all procs in env_comm
                action = env_comm.bcast(action, root=0)
                next_state, reward, done, _ = workflow.env.step(action)

                if ExaComm.env_comm.rank == 0:
                    total_reward += reward
                    memory = (current_state, action, reward,
                              next_state, done, total_reward)
                    batch_data = []
                    workflow.agent.remember(
                        memory[0], memory[1], memory[2], memory[3], memory[4])
                    # TODO: we need a memory class to scale
                    batch_data = next(workflow.agent.generate_data())
                    logger.info(
                        'Rank[{}] - Generated data: {}'.format(env_comm.rank, len(batch_data[0])))
                    try:
                        buffer_length = len(workflow.agent.memory)
                    except:
                        buffer_length = workflow.agent.replay_buffer.get_buffer_length()
                    logger.info(
                        'Rank[{}] - # Memories: {}'.format(env_comm.rank, buffer_length))

                # Learner
                if ExaComm.is_learner():
                    # Push memories to learner
                    train_return = workflow.agent.train(batch_data)
                    if train_return is not None:
                        # indices, loss = train_return
                        workflow.agent.set_priorities(*train_return)
                    workflow.agent.target_train()
                    rank0_epsilon = workflow.agent.epsilon

                if ExaComm.env_comm.rank == 0:
                    # Update state
                    current_state = next_state
                    logger.info('Rank[%s] - Total Reward:%s' %
                                (str(env_comm.rank), str(total_reward)))
                    steps += 1
                    if steps >= workflow.nsteps:
                        done = True

                    # Save memory for offline analysis
                    train_writer.writerow([time.time(), current_state, action, reward,
                                           next_state, total_reward, done, e, steps, policy_type, rank0_epsilon])
                    train_file.flush()

                # Broadcast done
                done = env_comm.bcast(done, 0)

            end_time_episode = time.time()
            if ExaComm.env_comm.rank == 0:
                logger.info('Rank[%s] run-time for episode %s: %s ' %
                            (str(env_comm.rank), str(e), str(end_time_episode - start_time_episode)))
                logger.info('Rank[%s] run-time for episode per step %s: %s '
                            % (str(env_comm.rank), str(e), str((end_time_episode - start_time_episode) / steps)))

        if ExaComm.env_comm.rank == 0:
            # Save Learning target model
            if ExaComm.is_learner():
                workflow.agent.save(workflow.results_dir + '/' + filename_prefix + '.h5')
            train_file.close()
