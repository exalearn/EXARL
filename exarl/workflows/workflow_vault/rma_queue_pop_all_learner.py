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

class RMA_QUEUE_POP_ALL(erl.ExaWorkflow):
    def __init__(self):
        print('Creating RMA_QUEUE_POP_ALL workflow...')

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
            # Initialize the queue data structure
            data_queue = MPI_RMA_QUEUE(agent_comm, mpi_settings.is_learner(), data=agent_batch, length=512, failPush=True, usePopAll=True)


        if mpi_settings.is_learner() and learner_comm.rank == 0:
            # Write target weight to model window of learner
            model_win.Lock(0)
            model_win.Put(serial_target_weights, target_rank=0)
            model_win.Unlock(0)

        # Synchronize
        agent_comm.Barrier()

        # Learner
        if mpi_settings.is_learner():
            start = MPI.Wtime()
            # set up a set data structure containing all the actors rank
            actor_numbers = agent_comm.size - learner_comm.size
            actor_ranks = set([_ for _ in range(learner_comm.size, agent_comm.size)])
            episode_count_learner = 0
            epsilon = np.array(workflow.agent.epsilon, dtype=np.float64)


            # initialize epsilon
            epsilon_win.Lock(0)
            epsilon_win.Put(epsilon, target_rank=0)
            epsilon_win.Unlock(0)

            next_s = 0
            req_successefully = False
            # request data from an actor
            while not req_successefully:
                next_s = random.sample(actor_ranks,1)[0]
                req_successefully = data_queue.request_pop_all(next_s)
            s = next_s

            while episode_count_learner < actor_numbers:
                # wait for getting data
                grouped_agent_data = data_queue.wait_pop_all(s)
                # directly request for new data
                if episode_count_learner < actor_numbers - 1 or not None in grouped_agent_data:
                    req_successefully = False
                    while not req_successefully:
                        next_s = random.sample(actor_ranks,1)[0]
                        req_successefully = data_queue.request_pop_all(next_s)

                # Train on previously requested data
                for agent_data in grouped_agent_data:
                    if agent_data is None : # the actor ended all episodes
                        episode_count_learner += 1
                        actor_ranks.remove(s)
                    else :
                        # Train
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
                # the next request become the current one
                s = next_s
        else:
            if mpi_settings.is_actor():
                # Setup logger
                filename_prefix = 'ExaLearner_' + 'Episodes%s_Steps%s_Rank%s_memory_v1' \
                    % (str(workflow.nepisodes), str(workflow.nsteps), str(agent_comm.rank))
                train_file = open(workflow.results_dir + '/' +
                                  filename_prefix + ".log", 'w')
                train_writer = csv.writer(train_file, delimiter=" ")

                # Constants and buffers
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

                episode_count_actor = env_comm.bcast(episode_count_actor, root=0)
                if episode_count_actor >= workflow.nepisodes:
                    break
                # Reset variables each episode
                workflow.env.seed(0)
                # TODO: optimize some of these variables out for env processes
                current_state = workflow.env.reset()
                total_rewards = 0
                steps = 0
                action = 0
                done = False
                buff = bytearray(serial_target_weights_size)

                # Steps in an episode
                while steps < workflow.nsteps:
                    logger.debug('ASYNC::run() agent_comm.rank{}; step({} of {})'
                                 .format(agent_comm.rank, steps, (workflow.nsteps - 1)))
                    if mpi_settings.is_actor():
                        # buffers
                        local_epsilon = np.array(workflow.agent.epsilon)

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

                        # Write data to queue
                        capacity, lost = data_queue.push(batch_data, agent_comm.rank)
                        while lost: # blocking wait if the queue is full
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


        if mpi_settings.is_agent():
            agent_comm.Barrier()
            episode_win.Free()
            epsilon_win.Free()
            indices_win.Free()
            loss_win.Free()
            model_win.Free()

        if mpi_settings.is_actor():
            logger.info(f'Agent[{agent_comm.rank}] timing info:\n')
            workflow.agent.print_timers()
