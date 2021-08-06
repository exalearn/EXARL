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
import sys
import pickle
import random
from exarl.network.data_structures import MPI_RMA_QUEUE
logger = log.setup_logger(__name__, cd.run_params['log_level'])

class ML_RMA_QUEUE(erl.ExaWorkflow):
    def __init__(self):
        print('Creating ML_RMA_QUEUE workflow ...')

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
            # -- Get size of a bitmap element (use BYTE here)
            disp = MPI.BYTE.Get_size()
            actors_ended_bitmap = None
            win_size = 0
            if mpi_settings.is_learner() and learner_comm.rank == 0:
                actors_number = agent_comm.size - learner_comm.size
                actors_ended_bitmap = np.zeros(actors_number, dtype=np.byte)
                win_size = disp*actors_number
            # Allocate actors_ended_bitma window
            actors_ended_bitmap_win = MPI.Win.Allocate(win_size, disp, comm=agent_comm)
            # Initialize actors_ended_bitma window
            if mpi_settings.is_learner() and learner_comm.rank == 0:
                actors_ended_bitmap_win.Lock(0)
                actors_ended_bitmap_win.Put(actors_ended_bitmap, target_rank=0)
                actors_ended_bitmap_win.Unlock(0)

            # -- Get size of episode counter
            disp = MPI.DOUBLE.Get_size()
            episode_data = None
            win_size = 0
            if mpi_settings.is_learner() and learner_comm.rank == 0:
                episode_data = np.zeros(1, dtype=np.float64)
                win_size = disp
            # Allocate episode window
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
            # Allocate epsilon window
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
            # Allocate indices window
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
            # Allocate loss window
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


            try:
                rma_queue_length = cd.run_params['rma_queue_length']
            except:
                rma_queue_length = 1024

            agent_batch = next(workflow.agent.generate_data())
            # Allocate data_queue
            data_queue = MPI_RMA_QUEUE(agent_comm, mpi_settings.is_learner(), data=agent_batch, length=rma_queue_length, failPush=True)


        if mpi_settings.is_learner() and learner_comm.rank == 0:
            # Write target weight to model window of learner
            model_win.Lock(0)
            model_win.Put(serial_target_weights, target_rank=0)
            model_win.Unlock(0)

        # Synchronize
        agent_comm.Barrier()

        # Learner
        if mpi_settings.is_learner():
            # initialize variables
            one = np.ones(1, dtype=np.byte)
            byte_size = MPI.BYTE.Get_size()
            total_actors_number = agent_comm.size - learner_comm.size
            actors_ended_bitmap = np.zeros(total_actors_number, dtype=np.byte)
            epsilon = np.array(workflow.agent.epsilon, dtype=np.float64)
            agent_data = None

            # compute the actors_umber per learner
            learner_actors_number = total_actors_number//learner_comm.size
            actor_rest = total_actors_number%learner_comm.size

            # distribute the rest of the actors amoung the learners
            if learner_comm.rank < actor_rest : # distribute the rest
                learner_actors_number += 1
                actor_rank_start = learner_comm.size + learner_actors_number*learner_comm.rank
                actor_rank_end   = actor_rank_start + learner_actors_number
            else:
                actor_rank_start = learner_comm.size + actor_rest + learner_actors_number*learner_comm.rank
                actor_rank_end   = actor_rank_start + learner_actors_number

            # create a local set data structure for each learner
            # at the begining each group of actors is assigned to one specific learner
            actor_ranks = set([_ for _ in range(actor_rank_start, actor_rank_end)])

            # initialize epsilon
            epsilon_win.Lock(0)
            epsilon_win.Put(epsilon, target_rank=0)
            epsilon_win.Unlock(0)

            # Main loop
            while True:

                ##########################
                # Get Data
                ##########################
                process_has_data = 1
                if len(actor_ranks) > 0:
                    pop_succes = False
                    while not pop_succes:
                        s = random.sample(actor_ranks, 1)[0] # get randomly an active actor
                        pop_succes, agent_data = data_queue.pop(s) # try to get data from the queue
                else : # there is no more active actors for the current learner
                    # get the shared bitmap of all active actors
                    actors_ended_bitmap_win.Lock(0)
                    #actors_ended_bitmap_win.Get_accumulate(one, actors_ended_bitmap, target_rank=0, op=MPI.NO_OP)
                    actors_ended_bitmap_win.Get(actors_ended_bitmap, target_rank=0)
                    actors_ended_bitmap_win.Unlock(0)

                    # compute the active actors
                    current_active_actors = np.where(actors_ended_bitmap == 0)[0] + learner_comm.size

                    if current_active_actors.size == 0 : # No more active actors
                        agent_data = [1] # dummy agent_data to pass the "if agent_data is None" test
                        process_has_data = 0 # stop training
                        #(we stop the training once there is no enough data to perform a new parallel horovod training)

                    else:
                        got_data = False
                        for active_actor in current_active_actors :
                            pop_succes, agent_data = data_queue.pop(active_actor)
                            if pop_succes :
                                if agent_data is None:
                                    data_queue.push(None, active_actor) # push back to notify the corresponding learner
                                else :
                                    got_data = True
                                    break # got data for training
                        if not got_data :
                         continue

                # Check if agent_data is none (None is used to inform the learner that an actor finished all episodes)
                if agent_data is None :
                    actor_ranks.remove(s)
                    actors_ended_bitmap_win.Lock(0)
                    actors_ended_bitmap_win.Accumulate(one, 0, target=[(s - learner_comm.size), byte_size], op=MPI.SUM)
                    actors_ended_bitmap_win.Unlock(0)
                    continue


                ##########################
                # Horovod train
                ##########################
                # Synchronize learners
                sum_process_has_data = learner_comm.allreduce(process_has_data, op=MPI.SUM)
                if (sum_process_has_data / learner_comm.size) < 1.0:
                    break # there is at least one learner withou data

                # Train
                train_return = workflow.agent.train(agent_data)
                if train_return is not None:
                    if not np.array_equal(train_return[0], (-1 * np.ones(workflow.agent.batch_size))):
                        indices, loss = train_return
                        indices = np.array(indices, dtype=np.intc)
                        loss = np.array(loss, dtype=np.float64)

                if learner_comm.rank == 0:
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
        else:
            if mpi_settings.is_actor():
                # Setup logger
                filename_prefix = 'ExaLearner_' + 'Episodes%s_Steps%s_Rank%s_memory_v1' \
                    % (str(workflow.nepisodes), str(workflow.nsteps), str(agent_comm.rank))
                train_file = open(workflow.results_dir + '/' +
                                  filename_prefix + ".log", 'w')
                train_writer = csv.writer(train_file, delimiter=" ")

                # constants and buffers
                episode_count_actor = np.zeros(1, dtype=np.float64)
                one = np.ones(1, dtype=np.float64)
                epsilon = np.zeros(1, dtype=np.float64)
                indices = -1 * np.ones(workflow.agent.batch_size, dtype=np.intc)
                indices_none = -1 * np.ones(workflow.agent.batch_size, dtype=np.intc)
                loss = np.zeros(workflow.agent.batch_size, dtype=np.float64)

                # Get initial value of episode counter
                episode_win.Lock(0, lock_type=MPI.LOCK_SHARED)
                # Atomic Get using Get_accumulate
                episode_win.Get_accumulate(one, episode_count_actor, target_rank=0, op=MPI.NO_OP)
                episode_win.Unlock(0)

            start = MPI.Wtime()
            while episode_count_actor < workflow.nepisodes:
                if mpi_settings.is_actor():
                    episode_win.Lock(0)
                    # Atomic Get_accumulate to increment the episode counter
                    episode_win.Get_accumulate(one, episode_count_actor, target_rank=0, op=MPI.SUM)
                    episode_win.Unlock(0)
                # Bcast the episode counter
                episode_count_actor = env_comm.bcast(episode_count_actor, root=0)
                if episode_count_actor >= workflow.nepisodes:
                    break

                # Reset variables each episode
                workflow.env.seed(0)
                # Variables and buffers
                current_state = workflow.env.reset()
                total_rewards = 0
                steps = 0
                action = 0
                done = False
                buff = bytearray(serial_target_weights_size)
                local_epsilon = np.array(workflow.agent.epsilon)

                # Steps in an episode
                while steps < workflow.nsteps:
                    logger.debug('ASYNC::run() agent_comm.rank{}; step({} of {})'
                                 .format(agent_comm.rank, steps, (workflow.nsteps - 1)))
                    if mpi_settings.is_actor():
                        # Get weights
                        model_win.Lock(0, lock_type=MPI.LOCK_SHARED)
                        model_win.Get(buff, target_rank=0)
                        model_win.Unlock(0)

                        # Get epsilon
                        epsilon_win.Lock(0)
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

                        # update the agent
                        workflow.agent.epsilon = min(epsilon, local_epsilon)
                        target_weights = MPI.pickle.loads(buff)
                        workflow.agent.set_weights(target_weights)
                        if not np.array_equal(indices, indices_none):
                            workflow.agent.set_priorities(indices, loss)

                        # inference action
                        if workflow.action_type == 'fixed':
                            action, policy_type = 0, -11
                        else:
                            action, policy_type = workflow.agent.action(
                                current_state)

                    # Broadcast action to all procs in env_comm
                    action = env_comm.bcast(action, root=0)

                    # Environment step
                    next_state, reward, done, _ = workflow.env.step(action)

                    if mpi_settings.is_actor():
                        # Save memory
                        total_rewards += reward
                        memory = (current_state, action, reward, next_state, done, total_rewards)
                        workflow.agent.remember(memory[0], memory[1], memory[2], memory[3], memory[4])
                        batch_data = next(workflow.agent.generate_data())

                        # Write data to the queue
                        capacity, lost = data_queue.push(batch_data, agent_comm.rank)
                        while lost: # try to push data until success
                            capacity, lost = data_queue.push(batch_data, agent_comm.rank)

                    if steps >= workflow.nsteps - 1:
                        done = True

                    if mpi_settings.is_actor():
                        train_writer.writerow([time.time(), current_state, action, reward, next_state, total_rewards,
                                               done, episode_count_actor[0], steps, policy_type, workflow.agent.epsilon])
                        train_file.flush()

                    # Update state and step
                    current_state = next_state
                    steps += 1

                    # Broadcast done
                    done = env_comm.bcast(done, root=0)
                    # Break for loop if done
                    if done:
                        break
            logger.info('Worker time = {}'.format(MPI.Wtime() - start))
            if mpi_settings.is_actor():
                train_file.close()

        # inform the learner
        if mpi_settings.is_actor():
            # "None" message inform the learner that the actor ended all episodes
            capacity, lost = data_queue.push(None, agent_comm.rank)
            while lost:
                capacity, lost = data_queue.push(None, agent_comm.rank)

        # Free ressources
        if mpi_settings.is_agent():
            agent_comm.Barrier()
            actors_ended_bitmap_win.Free()
            episode_win.Free()
            epsilon_win.Free()
            indices_win.Free()
            loss_win.Free()
            model_win.Free()

        if mpi_settings.is_actor():
            logger.info(f'Agent[{agent_comm.rank}] timing info:\n')
            workflow.agent.print_timers()
