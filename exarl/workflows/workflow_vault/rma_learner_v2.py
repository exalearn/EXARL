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
import exarl.mpi_settings as mpi_settings
import time
import csv
from mpi4py import MPI
import numpy as np
import exarl as erl
from exarl.utils.profile import *
import exarl.utils.log as log
import exarl.utils.candleDriver as cd

logger = log.setup_logger(__name__, cd.run_params['log_level'])

class RMA_ASYNC_v2(erl.ExaWorkflow):
    def __init__(self):
        print('Creating RMA async workflow...')

    @PROFILE
    def run(self, workflow):
        total_comm_time = 0.0
        # MPI communicators
        agent_comm = mpi_settings.agent_comm
        env_comm = mpi_settings.env_comm
        learner_comm = mpi_settings.learner_comm

        if mpi_settings.is_learner():
            workflow.agent.set_learner()

        # Allocate RMA windows
        if mpi_settings.is_agent():
            # -- Get size of episode counter
            disp = MPI.DOUBLE.Get_size()
            episode_data = None
            win_size = 0
            if mpi_settings.is_learner() and learner_comm.rank == 0:
                episode_data = np.zeros(1, dtype=np.float64)
                win_size = disp
            # Create episode window (attach instead of allocate for zero initialization)
            episode_win = MPI.Win.Allocate(win_size, disp, comm=agent_comm)
            # Initialize episode window
            if mpi_settings.is_learner() and learner_comm.rank == 0:
                episode_win.Lock(0)
                episode_win.Put(episode_data, target_rank=0)
                episode_win.Unlock(0)

            # -- Get size of epsilon
            disp = MPI.DOUBLE.Get_size()
            epsilon = None
            win_size = 0
            if mpi_settings.is_learner() and learner_comm.rank == 0:
                epsilon = np.zeros(1, dtype=np.float64)
                win_size = disp
            # Create epsilon window
            epsilon_win = MPI.Win.Allocate(win_size, disp, comm=agent_comm)
            if mpi_settings.is_learner() and learner_comm.rank == 0:
                epsilon_win.Lock(0)
                epsilon_win.Put(epsilon, target_rank=0)
                epsilon_win.Unlock(0)

            # -- Get size of individual indices
            disp = MPI.INT.Get_size()
            indices = None
            win_size = 0
            if mpi_settings.is_learner() and learner_comm.rank == 0:
                indices = -1 * np.ones(workflow.agent.batch_size, dtype=np.intc)
                win_size = workflow.agent.batch_size*disp
            # Create indices window
            indices_win = MPI.Win.Allocate(win_size, disp, comm=agent_comm)
            # Initialize indices window
            if mpi_settings.is_learner() and learner_comm.rank == 0:
                indices_win.Lock(0)
                indices_win.Put(indices, target_rank=0)
                indices_win.Unlock(0)

            # -- Get size of loss
            disp = MPI.DOUBLE.Get_size()
            loss = None
            win_size = 0
            if mpi_settings.is_learner() and learner_comm.rank == 0:
                loss = np.zeros(workflow.agent.batch_size, dtype=np.float64)
                win_size = workflow.agent.batch_size*disp
            # Create loss window
            loss_win = MPI.Win.Allocate(win_size, disp, comm=agent_comm)
            # Initialize loss window
            if mpi_settings.is_learner() and learner_comm.rank == 0:
                temp =  np.zeros(workflow.agent.batch_size, dtype=np.float64)
                loss_win.Lock(0)
                loss_win.Put(temp, target_rank=0)
                loss_win.Unlock(0)


            # Get serialized target weights size
            target_weights = workflow.agent.get_weights()
            serial_target_weights = MPI.pickle.dumps(target_weights)
            serial_target_weights_size = len(serial_target_weights)
            target_weights_size = 0
            if mpi_settings.is_learner():
                target_weights_size = serial_target_weights_size
            # Allocate model window
            model_win = MPI.Win.Allocate(target_weights_size, 1, comm=agent_comm)

            # Get serialized batch data size
            agent_batch = next(workflow.agent.generate_data())
            serial_agent_batch = (MPI.pickle.dumps(agent_batch))
            serial_agent_batch_size = len(serial_agent_batch)
            nserial_agent_batch = 0
            if mpi_settings.is_actor():
                nserial_agent_batch = serial_agent_batch_size
            # Allocate data window
            data_win = MPI.Win.Allocate(nserial_agent_batch, 1, comm=agent_comm)

        if mpi_settings.is_learner() and learner_comm.rank == 0:
            # Write target weight to model window of learner
            model_win.Lock(0)
            model_win.Put(serial_target_weights, target_rank=0)
            model_win.Unlock(0)

        # Synchronize
        agent_comm.Barrier()

        # Learner
        if mpi_settings.is_learner():
            # Initialize batch data buffer
            data_buffer = bytearray(serial_agent_batch_size)
            episode_count_learner = np.zeros(1, dtype=np.float64)
            epsilon = np.array(workflow.agent.epsilon, dtype=np.float64)
            #learner_counter = 0
            # Initialize epsilon
            epsilon_win.Lock(0)
            epsilon_win.Put(epsilon, target_rank=0)
            epsilon_win.Unlock(0)

            while episode_count_learner < workflow.nepisodes:
                # Check episode counter
                episode_win.Lock(0, lock_type=MPI.LOCK_SHARED)
                # Atomic Get_accumulate to fetch episode count
                episode_win.Get_accumulate(np.ones(1, dtype=np.float64), episode_count_learner, target_rank=0, op=MPI.NO_OP)
                episode_win.Unlock(0)

                # Go randomly over all actors
                s = np.random.randint(low=learner_comm.size, high=agent_comm.size)
                # Get data
                data_win.Lock(s, lock_type=MPI.LOCK_SHARED)
                data_win.Get(data_buffer, target_rank=s, target=None)
                data_win.Unlock(s)

                # Continue to other actor if data_buffer is empty
                try:
                    agent_data = MPI.pickle.loads(data_buffer)
                except:
                    continue


                # Train & Target train
                #print("--------------------- Train ...",learner_comm.rank, " episode : ", episode_count_learner)
                train_return = workflow.agent.train(agent_data)
                if train_return is not None:
                    if not np.array_equal(train_return[0], (-1 * np.ones(workflow.agent.batch_size))):
                        indices, loss = train_return
                        indices = np.array(indices, dtype=np.intc)
                        loss = np.array(loss, dtype=np.float64)

                # Write indices to memory pool
                indices_win.Lock(0)
                indices_win.Put(indices, target_rank=0)
                indices_win.Unlock(0)

                # Write losses to memory pool
                loss_win.Lock(0)
                loss_win.Put(loss, target_rank=0)
                loss_win.Unlock(0)

                # TODO: Double check if this is already in the DQN code
                workflow.agent.target_train()
                # Share new model weights
                target_weights = workflow.agent.get_weights()
                serial_target_weights = MPI.pickle.dumps(target_weights)
                model_win.Lock(0)
                model_win.Put(serial_target_weights, target_rank=0)
                model_win.Unlock(0)
                #learner_counter += 1

            logger.info('Learner exit on rank_episode: {}_{}'.format(agent_comm.rank, episode_data))

        # Actors
        else:
            local_actor_episode_counter = 0
            if mpi_settings.is_actor():
                # Logging files
                filename_prefix = 'ExaLearner_' + 'Episodes%s_Steps%s_Rank%s_memory_v1' \
                    % (str(workflow.nepisodes), str(workflow.nsteps), str(agent_comm.rank))
                train_file = open(workflow.results_dir + '/' + filename_prefix + ".log", 'w')
                train_writer = csv.writer(train_file, delimiter=" ")

                episode_count_actor = np.zeros(1, dtype=np.float64)
                one = np.ones(1, dtype=np.float64)
                epsilon = np.zeros(1, dtype=np.float64)
                indices = -1 * np.ones(workflow.agent.batch_size, dtype=np.int32)
                loss = np.zeros(workflow.agent.batch_size, dtype=np.float64)

                # Get initial value of episode counter
                episode_win.Lock(0, lock_type=MPI.LOCK_SHARED)
                # Atomic Get using Get_accumulate
                episode_win.Get_accumulate(one, episode_count_actor, target_rank=0, op=MPI.NO_OP)
                episode_win.Unlock(0)


            while episode_count_actor < workflow.nepisodes:
                if mpi_settings.is_actor():
                    episode_win.Lock(0)
                    # Atomic Get_accumulate to increment the episode counter
                    episode_win.Get_accumulate(one, episode_count_actor, target_rank=0, op=MPI.SUM)
                    episode_win.Unlock(0)

                episode_count_actor = env_comm.bcast(episode_count_actor, root=0)
                #print("------------- actor ", episode_count_actor)

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
                    if mpi_settings.is_actor():

                        # buffers
                        buff = bytearray(serial_target_weights_size)
                        local_epsilon = np.array(workflow.agent.epsilon)

                        total_comm_time -= MPI.Wtime()
                        # Update model weight
                        # TODO: weights are updated each step -- REVIEW --
                        model_win.Lock(0, lock_type=MPI.LOCK_SHARED)
                        model_win.Get(buff, target_rank=0)
                        model_win.Unlock(0)

                        # Get epsilon
                        epsilon_win.Lock(0, lock_type=MPI.LOCK_SHARED)
                        epsilon_win.Get_accumulate(local_epsilon, epsilon, target_rank=0, op=MPI.MIN)
                        epsilon_win.Unlock(0)

                        # Get indices
                        indices_win.Lock(0, lock_type=MPI.LOCK_SHARED)
                        indices_win.Get(indices, target_rank=0)
                        indices_win.Unlock(0)

                        # Get losses
                        loss_win.Lock(0, lock_type=MPI.LOCK_SHARED)
                        loss_win.Get(loss, target_rank=0)
                        loss_win.Unlock(0)

                        total_comm_time += MPI.Wtime()

                        # Update workflow
                        target_weights = MPI.pickle.loads(buff)
                        workflow.agent.set_weights(target_weights)
                        workflow.agent.epsilon = min(epsilon, local_epsilon)

                        if not np.array_equal(indices, (-1 * np.ones(workflow.agent.batch_size, dtype=np.intc))):
                            workflow.agent.set_priorities(indices, loss)

                        # Inference action
                        if workflow.action_type == 'fixed':
                            action, policy_type = 0, -11
                        else:
                            action, policy_type = workflow.agent.action(current_state)

                    # Broadcast action to all procs in env_comm
                    action = env_comm.bcast(action, root=0)

                    # Environment step
                    next_state, reward, done, _ = workflow.env.step(action)

                    steps += 1
                    if steps >= workflow.nsteps:
                        done = True
                    # Broadcast done
                    done = env_comm.bcast(done, root=0)

                    if mpi_settings.is_actor():
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

        if mpi_settings.is_agent():
            agent_comm.Barrier()
            episode_win.Free()
            epsilon_win.Free()
            indices_win.Free()
            loss_win.Free()
            model_win.Free()
            data_win.Free()

            aggregate_comm_time = np.zeros(1, np.float64)
            total_time_buf= total_comm_time*np.ones(1, np.float64)
            agent_comm.Reduce(total_time_buf, aggregate_comm_time, op=MPI.SUM, root=0)

            if mpi_settings.is_actor():
                print("[{}] Total communication time (Get()) : {} s ".format(agent_comm.rank, total_comm_time))

            if agent_comm.rank == 0 :
                print("Total aggregated communication time : {} s. Average time : {} s".format(aggregate_comm_time[0],aggregate_comm_time[0]/(agent_comm.size - learner_comm.size)))
