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
import numpy as np
from tensorflow.python.ops.gen_batch_ops import batch
import exarl as erl
from exarl.utils.introspect import *
from exarl.utils.profile import *
import exarl.utils.log as log
import exarl.utils.candleDriver as cd
from exarl.base.comm_base import ExaComm
from exarl.network.data_structures import *
from exarl.network.simple_comm import ExaSimple
MPI = ExaSimple.MPI

logger = log.setup_logger(__name__, cd.run_params['log_level'])

class RMA(erl.ExaWorkflow):
    def __init__(self):
        print("Creating RMA workflow")
        data_exchange_constructors = {
            "buff_unchecked": ExaMPIBuffUnchecked,
            "buff_checked": ExaMPIBuffChecked,
            "queue_distribute": ExaMPIDistributedQueue,
            "stack_distribute": ExaMPIDistributedStack,
            "queue_central": ExaMPICentralizedQueue,
            "stack_central": ExaMPICentralizedStack
        }
        # target weights - This should be an unchecked buffer that will always succed a pop since weight need to be shared with everyone
        self.target_weight_data_structure = data_exchange_constructors[cd.lookup_params('target_weight_structure', default='buff_unchecked')]

        # Batch data
        self.batch_data_structure = data_exchange_constructors[cd.lookup_params('data_structure', default='buff_unchecked')]
        self.de_length = cd.lookup_params('data_structure_length', default=32)
        self.de_lag = None  # cd.lookup_params('max_model_lag')
        logger.info("Creating RMA data exchange workflow", cd.lookup_params('data_structure',
                                                                            default='buff_unchecked'), "length", self.de_length, "lag", self.de_lag)

        # Loss and indicies
        self.de = cd.lookup_params('loss_data_structure', default='buff_unchecked')
        self.ind_loss_data_structure = data_exchange_constructors[cd.lookup_params('loss_data_structure', default='buff_unchecked')]
        logger.info('Creating RMA loss exchange workflow with ', self.de)

        priority_scale = cd.run_params['priority_scale']
        self.use_priority_replay = (priority_scale is not None and priority_scale > 0)

    @PROFILE
    def run(self, workflow):
        # Number of learner processes
        num_learners = ExaComm.num_learners

        # MPI communicators
        agent_comm = ExaComm.agent_comm
        env_comm = ExaComm.env_comm
        if ExaComm.is_learner():
            learner_comm = ExaComm.learner_comm

        # Allocate RMA windows
        if ExaComm.is_agent():
            episode_const = ExaMPIConstant(agent_comm, ExaComm.is_learner() and learner_comm.rank == 0, np.int64, name="Episode_Const")
            epsilon_const = ExaMPIConstant(agent_comm, ExaComm.is_learner() and learner_comm.rank == 0, np.float64, name="Epsilon_Const")

            if self.use_priority_replay:
                # Create windows for priority replay (loss and indicies)
                indices_for_size = -1 * np.ones(workflow.agent.batch_size, dtype=np.intc)
                loss_for_size = np.zeros(workflow.agent.batch_size, dtype=np.float64)
                indicies_and_loss_for_size = (indices_for_size, loss_for_size)
                ind_loss_buffer = self.ind_loss_data_structure(agent_comm, indicies_and_loss_for_size, rank_mask=ExaComm.is_actor(),
                                                               length=num_learners, max_model_lag=None, name="Loss_Buffer")

            # Get serialized target weights size
            learner_counter = np.int64(0)
            target_weights = (workflow.agent.get_weights(), learner_counter)
            model_buff = self.target_weight_data_structure(agent_comm, target_weights, rank_mask=ExaComm.is_learner() and
                                                           learner_comm.rank == 0,  length=1, max_model_lag=None, failPush=False, name="Model_Buffer")

            # Get serialized batch data size
            learner_counter = np.int64(0)
            agent_batch = (next(workflow.agent.generate_data()), learner_counter)
            batch_data_exchange = self.batch_data_structure(agent_comm, agent_batch, rank_mask=ExaComm.is_actor(),
                                                            length=self.de_length, max_model_lag=self.de_lag, name="Data_Exchange")
            # This is a data/flag that lets us know we have data
            agent_data = None

        # Synchronize
        agent_comm.barrier()

        # Learner
        if ExaComm.is_learner():
            # Initialize counter
            episode_count_learner = 0
            if learner_comm.rank == 0:
                epsilon_const.put(workflow.agent.epsilon, 0)

            while True:
                # Define flags to keep track of data
                process_has_data = 0
                sum_process_has_data = 0

                if learner_comm.rank == 0:
                    episode_count_learner = episode_const.get(0)

                if num_learners > 1:
                    episode_count_learner = learner_comm.bcast(episode_count_learner, root=0)

                if episode_count_learner >= workflow.nepisodes:
                    break

                if agent_data is None:
                    agent_data, actor_idx, actor_counter = batch_data_exchange.get_data(
                        learner_counter, learner_comm.size, agent_comm.size)
                    ib.update("RMA_Learner_Pop_Data", 1)
                    ib.simpleTrace("RMA_Learner_Get_Data", actor_idx, actor_counter, learner_counter - actor_counter, 0)

                # Check the data_buffer again if it is empty
                if agent_data is not None:
                    process_has_data = 1

                # Do an allreduce to check if all learners have data
                sum_process_has_data = learner_comm.allreduce(process_has_data, op=MPI.SUM)
                if sum_process_has_data < learner_comm.size:
                    continue

                train_return = workflow.agent.train(agent_data)
                ib.update("RMA_Learner_Train", 1)

                if self.use_priority_replay and train_return is not None:
                    if not np.array_equal(train_return[0], (-1 * np.ones(workflow.agent.batch_size))):
                        indices, loss = train_return
                        indices = np.array(indices, dtype=np.intc)
                        loss = np.array(loss, dtype=np.float64)
                        # Write indices to memory pool
                        ind_loss_buffer.push((indices, loss), rank=actor_idx)
                        ib.update("RMA_Learner_Loss_Push", 1)
                learner_counter += 1
                agent_data = None

                if ExaComm.is_learner() and learner_comm.rank == 0:
                    workflow.agent.target_train()
                    ib.update("RMA_Learner_Target_Train", 1)
                    # Share new model weights
                    target_weights = (workflow.agent.get_weights(), learner_counter)
                    model_buff.push(target_weights, rank=0)
                    ib.update("RMA_Learner_Model_Push", 1)

            logger.info('Learner exit on rank_episode: {}_{}'.format(agent_comm.rank, episode_count_learner))

        # Actors
        else:
            local_actor_episode_counter = 0
            if env_comm.rank == 0:
                # Logging files
                filename_prefix = 'ExaLearner_' + 'Episodes%s_Steps%s_Rank%s_memory_v1' \
                    % (str(workflow.nepisodes), str(workflow.nsteps), str(agent_comm.rank))
                train_file = open(workflow.results_dir + '/' + filename_prefix + ".log", 'w')
                train_writer = csv.writer(train_file, delimiter=" ")

            while True:
                if env_comm.rank == 0:
                    episode_count_actor = episode_const.inc(0)

                episode_count_actor = env_comm.bcast(episode_count_actor, root=0)

                # Check if we are done based on the completed episodes
                if episode_count_actor >= workflow.nepisodes:
                    break
                logger.info('Rank[{}] - working on episode: {}'.format(agent_comm.rank, episode_count_actor))

                # Episode initialization
                workflow.env.seed(0)
                current_state = workflow.env.reset()
                total_rewards = 0
                steps = 0
                action = 0
                done = False
                local_actor_episode_counter += 1

                while done != True:
                    if env_comm.rank == 0:
                        # Update model weight
                        target_weights, learner_counter = model_buff.pop(0)
                        ib.update("RMA_Env_Model_Pop", 1)
                        workflow.agent.set_weights(target_weights)
                        ib.simpleTrace("RMA_Actor_Get_Model", local_actor_episode_counter, learner_counter, 0, 0)

                        workflow.agent.epsilon = epsilon_const.min(workflow.agent.epsilon, 0)

                        if self.use_priority_replay:
                            # Get indices and losses
                            loss_data = ind_loss_buffer.pop(agent_comm.rank)
                            ib.update("RMA_Env_Loss_Pop", 1)
                            # print("loss data = ", loss_data, flush=True)
                            if loss_data is not None:
                                indices, loss = loss_data
                                if self.de == "buff_unchecked":
                                    if not np.array_equal(indices, (-1 * np.ones(workflow.agent.batch_size, dtype=np.intc))):
                                        workflow.agent.set_priorities(indices, loss)
                                else:
                                    workflow.agent.set_priorities(indices, loss)

                        # Inference action
                        action, policy_type = workflow.agent.action(current_state)
                        ib.update("RMA_Env_Inference", 1)
                        if workflow.action_type == 'fixed':
                            action, policy_type = 0, -11

                        # Broadcast episode count to all procs in env_comm
                        action = env_comm.bcast(action, 0)

                    # Environment step
                    ib.startTrace("step", 0)
                    next_state, reward, done, _ = workflow.env.step(action)
                    ib.stopTrace()
                    ib.update("RMA_Env_Step", 1)
                    ib.simpleTrace("RMA_Reward", steps, 1 if done else 0, local_actor_episode_counter, reward)

                    # Update current state and steps count
                    current_state = next_state
                    steps += 1
                    if steps >= workflow.nsteps:
                        done = True
                    # Broadcast done
                    done = env_comm.bcast(done, 0)

                    if env_comm.rank == 0:
                        # Save memory
                        total_rewards += reward
                        workflow.agent.remember(current_state, action, reward, next_state, done)
                        if workflow.agent.has_data():
                            batch_data = (next(workflow.agent.generate_data()), learner_counter)
                            ib.update("RMA_Env_Generate_Data", 1)
                            # Write to data window
                            capacity, lost = batch_data_exchange.push(batch_data)
                            ib.update("RMA_Env_Push_Data", 1)
                            ib.simpleTrace("RMA_Actor_Put_Data", capacity, lost, 0, 0)

                        # Log state, action, reward, ...
                        train_writer.writerow([time.time(), current_state, action, reward, next_state, total_rewards,
                                               done, local_actor_episode_counter, steps, policy_type, workflow.agent.epsilon])
                        train_file.flush()
                ib.update("RMA_Env_Episode", 1)

        # mpi4py may miss MPI Finalize sometimes -- using a barrier
        agent_comm.barrier()
