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
from exarl.base.comm_base import ExaComm
from exarl.network.simple_comm import ExaSimple
import time
import csv
import numpy as np
import exarl as erl
from exarl.utils.profile import *
import exarl.utils.log as log
import exarl.utils.candleDriver as cd
MPI = ExaSimple.MPI
logger = log.setup_logger(__name__, cd.run_params['log_level'])

class RMA(erl.ExaWorkflow):
    def __init__(self):
        print("Creating ML_RMA workflow")
        priority_scale = cd.run_params['priority_scale']
        self.use_priority_replay = (priority_scale is not None and priority_scale > 0)

    @PROFILE
    def run(self, workflow):
        # Number of learner processes
        num_learners = ExaComm.num_learners

        # MPI communicators
        agent_comm = ExaComm.agent_comm.raw()
        env_comm = ExaComm.env_comm.raw()
        if ExaComm.is_learner():
            learner_comm = ExaComm.learner_comm.raw()

        # Allocate RMA windows
        if ExaComm.is_agent():
            # Get size of episode counter
            disp = MPI.DOUBLE.Get_size()
            episode_data = None
            if ExaComm.is_learner() and learner_comm.rank == 0:
                episode_data = np.zeros(1, dtype=np.float64)
            # Create episode window (attach instead of allocate for zero initialization)
            episode_win = MPI.Win.Create(episode_data, disp, comm=agent_comm)

            # Get size of epsilon
            disp = MPI.DOUBLE.Get_size()
            epsilon = None
            if ExaComm.is_learner() and learner_comm.rank == 0:
                epsilon = np.zeros(1, dtype=np.float64)
            # Create epsilon window
            epsilon_win = MPI.Win.Create(epsilon, disp, comm=agent_comm)

            if self.use_priority_replay:
                # Get size of individual indices
                disp = MPI.INT.Get_size()
                indices = None
                if ExaComm.is_actor:
                    indices = -1 * np.ones(workflow.agent.batch_size, dtype=np.intc)
                # Create indices window
                indices_win = MPI.Win.Create(indices, disp, comm=agent_comm)

                # Get size of loss
                disp = MPI.DOUBLE.Get_size()
                loss = None
                if ExaComm.is_actor():
                    loss = np.zeros(workflow.agent.batch_size, dtype=np.float64)
                # Create epsilon window
                loss_win = MPI.Win.Create(loss, disp, comm=agent_comm)

            # Get serialized target weights size
            target_weights = workflow.agent.get_weights()
            serial_target_weights = MPI.pickle.dumps(target_weights)
            serial_target_weights_size = len(serial_target_weights)
            target_weights_size = 0
            if ExaComm.is_learner() and learner_comm.rank == 0:
                target_weights_size = serial_target_weights_size
            # Allocate model window
            model_win = MPI.Win.Allocate(target_weights_size, 1, comm=agent_comm)

            # Get serialized batch data size
            agent_batch = next(workflow.agent.generate_data())
            serial_agent_batch = (MPI.pickle.dumps(agent_batch))
            serial_agent_batch_size = len(serial_agent_batch)
            nserial_agent_batch = 0
            if ExaComm.is_actor():
                nserial_agent_batch = serial_agent_batch_size
            # Allocate data window
            data_win = MPI.Win.Allocate(nserial_agent_batch, 1, comm=agent_comm)

        if ExaComm.is_learner() and learner_comm.rank == 0:
            # Write target weight to model window of learner
            model_win.Lock(0)
            model_win.Put(serial_target_weights, target_rank=0)
            model_win.Unlock(0)

        # Synchronize
        agent_comm.Barrier()

        # Learner
        if ExaComm.is_learner():
            # Initialize batch data buffer
            data_buffer = bytearray(serial_agent_batch_size)
            episode_count_learner = np.zeros(1, dtype=np.float64)
            epsilon = np.array(workflow.agent.epsilon, dtype=np.float64)
            # learner_counter = 0
            # Initialize epsilon
            if learner_comm.rank == 0:
                epsilon_win.Lock(0)
                epsilon_win.Put(epsilon, target_rank=0)
                epsilon_win.Flush(0)
                epsilon_win.Unlock(0)

            while True:
                # Define flags to keep track of data
                process_has_data = 0
                sum_process_has_data = 0

                if learner_comm.rank == 0:
                    # Check episode counter
                    episode_win.Lock(0)
                    # Atomic Get_accumulate to fetch episode count
                    episode_win.Get_accumulate(np.ones(1, dtype=np.float64), episode_count_learner, target_rank=0, op=MPI.NO_OP)
                    episode_win.Flush(0)
                    episode_win.Unlock(0)

                if num_learners > 1:
                    episode_count_learner = learner_comm.bcast(episode_count_learner, root=0)

                if episode_count_learner >= workflow.nepisodes:
                    break

                # Randomly select actor
                low = learner_comm.size  # start
                high = agent_comm.size  # stop + 1
                actor_idx = np.random.randint(low=low, high=high, size=1)
                # Get data
                data_win.Lock(actor_idx)
                data_win.Get(data_buffer, target_rank=actor_idx, target=None)
                data_win.Unlock(actor_idx)

                # Check the data_buffer again if it is empty
                try:
                    agent_data = MPI.pickle.loads(data_buffer)
                    process_has_data = 1
                except:
                    logger.info('Data buffer is empty, continuing...')

                # Do an allreduce to check if all learners have data
                sum_process_has_data = learner_comm.allreduce(process_has_data, op=MPI.SUM)
                if sum_process_has_data < learner_comm.size:
                    continue

                # Train & Target train
                train_return = workflow.agent.train(agent_data)

                if self.use_priority_replay and train_return is not None:
                    if not np.array_equal(train_return[0], (-1 * np.ones(workflow.agent.batch_size))):
                        indices, loss = train_return
                        indices = np.array(indices, dtype=np.intc)
                        loss = np.array(loss, dtype=np.float64)

                        if ExaComm.is_learner() and learner_comm.rank == 0:
                            # Write indices to memory pool
                            indices_win.Lock(actor_idx)
                            indices_win.Put(indices, target_rank=actor_idx)
                            indices_win.Unlock(actor_idx)

                            # Write losses to memory pool
                            loss_win.Lock(actor_idx)
                            loss_win.Put(loss, target_rank=actor_idx)
                            loss_win.Unlock(actor_idx)

                if ExaComm.is_learner() and learner_comm.rank == 0:
                    # Target train
                    workflow.agent.target_train()
                    # Share new model weights
                    target_weights = workflow.agent.get_weights()
                    serial_target_weights = MPI.pickle.dumps(target_weights)
                    model_win.Lock(0)
                    model_win.Put(serial_target_weights, target_rank=0)
                    model_win.Unlock(0)
                # learner_counter += 1

            logger.info('Learner exit on rank_episode: {}_{}'.format(agent_comm.rank, episode_data))

        # Actors
        else:
            local_actor_episode_counter = 0
            if ExaComm.env_comm.rank == 0:
                # Logging files
                filename_prefix = 'ExaLearner_' + 'Episodes%s_Steps%s_Rank%s_memory_v1' \
                    % (str(workflow.nepisodes), str(workflow.nsteps), str(agent_comm.rank))
                train_file = open(workflow.results_dir + '/' + filename_prefix + ".log", 'w')
                train_writer = csv.writer(train_file, delimiter=" ")

                episode_count_actor = np.zeros(1, dtype=np.float64)
                one = np.ones(1, dtype=np.float64)
                epsilon = np.array(workflow.agent.epsilon, dtype=np.float64)
                if self.use_priority_replay:
                    indices = -1 * np.ones(workflow.agent.batch_size, dtype=np.int32)
                    loss = np.zeros(workflow.agent.batch_size, dtype=np.float64)

                # Get initial value of episode counter
                episode_win.Lock(0)
                # Atomic Get using Get_accumulate
                episode_win.Get_accumulate(one, episode_count_actor, target_rank=0, op=MPI.NO_OP)
                episode_win.Flush(0)
                episode_win.Unlock(0)

            while True:
                if ExaComm.env_comm.rank == 0:
                    episode_win.Lock(0)
                    # Atomic Get_accumulate to increment the episode counter
                    episode_win.Get_accumulate(one, episode_count_actor, target_rank=0)
                    episode_win.Flush(0)
                    episode_win.Unlock(0)

                episode_count_actor = env_comm.bcast(episode_count_actor, root=0)

                # Include another check to avoid each actor running extra
                # set of steps while terminating
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
                    if ExaComm.env_comm.rank == 0:
                        # Update model weight
                        buff = bytearray(serial_target_weights_size)
                        model_win.Lock(0)
                        model_win.Get(buff, target=0, target_rank=0)
                        model_win.Flush(0)
                        model_win.Unlock(0)
                        target_weights = MPI.pickle.loads(buff)
                        workflow.agent.set_weights(target_weights)

                        # Get epsilon
                        local_epsilon = np.array(workflow.agent.epsilon)
                        epsilon_win.Lock(0)
                        epsilon_win.Get_accumulate(local_epsilon, epsilon, target_rank=0, op=MPI.MIN)
                        epsilon_win.Flush(0)
                        epsilon_win.Unlock(0)

                        # Update the agent epsilon
                        workflow.agent.epsilon = min(epsilon, local_epsilon)

                        if self.use_priority_replay:
                            # Get indices
                            indices_win.Lock(agent_comm.rank)
                            indices_win.Get(indices, target_rank=agent_comm.rank)
                            indices_win.Flush(agent_comm.rank)
                            indices_win.Unlock(agent_comm.rank)

                            # Get losses
                            loss_win.Lock(agent_comm.rank)
                            loss_win.Get(loss, target_rank=agent_comm.rank)
                            loss_win.Flush(agent_comm.rank)
                            loss_win.Unlock(agent_comm.rank)

                            if not np.array_equal(indices, (-1 * np.ones(workflow.agent.batch_size, dtype=np.intc))):
                                workflow.agent.set_priorities(indices, loss)

                        # Inference action
                        if workflow.action_type == 'fixed':
                            action, policy_type = 0, -11
                        else:
                            action, policy_type = workflow.agent.action(current_state)

                        # Broadcast episode count to all procs in env_comm
                        action = env_comm.bcast(action, root=0)

                    # Environment step
                    next_state, reward, done, _ = workflow.env.step(action)

                    steps += 1
                    if steps >= workflow.nsteps:
                        done = True
                    # Broadcast done
                    done = env_comm.bcast(done, root=0)

                    if ExaComm.env_comm.rank == 0:
                        # Save memory
                        total_rewards += reward
                        memory = (current_state, action, reward, next_state, done, total_rewards)
                        workflow.agent.remember(memory[0], memory[1], memory[2], memory[3], memory[4])

                        batch_data = next(workflow.agent.generate_data())
                        # Write to data window
                        serial_agent_batch = (MPI.pickle.dumps(batch_data))
                        data_win.Lock(agent_comm.rank)
                        data_win.Put(serial_agent_batch, target_rank=agent_comm.rank)
                        data_win.Unlock(agent_comm.rank)

                        # Log state, action, reward, ...
                        train_writer.writerow([time.time(), current_state, action, reward, next_state, total_rewards,
                                               done, local_actor_episode_counter, steps, policy_type, workflow.agent.epsilon])
                        train_file.flush()

        if ExaComm.is_agent():
            model_win.Free()
            data_win.Free()
