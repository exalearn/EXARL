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
import pickle


class ASYNC_v2(erl.ExaWorkflow):
    def __init__(self):
        print('Creating ASYNC learner workflow...')

    @PROFILE
    def run(self, workflow):

        # MPI communicators
        agent_comm = mpi_settings.agent_comm
        env_comm = mpi_settings.env_comm
        learner_comm = mpi_settings.learner_comm

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
                loss_win.Lock(0)
                loss_win.Put(loss, target_rank=0)
                loss_win.Unlock(0)

            # -- Get serialized target weights size
            target_weights = workflow.agent.get_weights()
            serial_target_weights = MPI.pickle.dumps(target_weights)
            serial_target_weights_size = len(serial_target_weights)
            target_weights_size = 0
            if mpi_settings.is_learner():
                target_weights_size = serial_target_weights_size
            # Allocate model window
            model_win = MPI.Win.Allocate(target_weights_size, 1, comm=agent_comm)
            # Initialize model window
            if mpi_settings.is_learner() and learner_comm.rank == 0:
                # Write target weight to model window of learner
                model_win.Lock(0)
                model_win.Put(serial_target_weights, target_rank=0)
                model_win.Unlock(0)

        # Set target model
        if mpi_settings.is_learner():
            workflow.agent.set_learner()
            workflow.agent.set_weights(target_weights)


        # Variables for all
        episode = 0
        episode_done = 0
        episode_interim = 0

        # temp vars
        local_episode = 0
        waiting_start = 0
        waiting_end = 0

        # Round-Robin Scheduler
        if mpi_settings.is_learner():
            start = MPI.Wtime()
            # worker_episodes = np.linspace(0, agent_comm.size - 2, agent_comm.size - 1)
            worker_episodes = np.arange(1, agent_comm.size)
            logger.debug('worker_episodes:{}'.format(worker_episodes))

            logger.info("Initializing ...\n")

            init_nepisodes = episode
            logger.debug('init_nepisodes:{}'.format(init_nepisodes))

            logger.debug("Continuing ...\n")
            actors_done = 0
            total_actors = agent_comm.size - learner_comm.size
            while actors_done < total_actors:
                # print("Running scheduler/learner episode: {}".format(episode))

                # Receive the rank of the worker ready for more work
                recv_data = agent_comm.recv(source=MPI.ANY_SOURCE)
                if recv_data[0]  == -1:
                    actors_done += 1
                    continue

                batch = recv_data[1]
                policy_type = recv_data[2]

                # Train
                train_return = workflow.agent.train(batch)
                if train_return is not None:
                    if not np.array_equal(train_return[0], (-1 * np.ones(workflow.agent.batch_size))):
                        indices, loss = train_return

                # TODO: Double check if this is already in the DQN code
                workflow.agent.target_train()
                if policy_type == 0:
                    workflow.agent.epsilon_adj()
                epsilon = workflow.agent.epsilon

                # save weights
                with open('target_weights.pkl', 'wb') as f:
                    pickle.dump(target_weights, f)

                # Send target weights
                logger.debug('rank0_epsilon:{}'.format(epsilon))

                target_weights = workflow.agent.get_weights()
                serial_target_weights = MPI.pickle.dumps(target_weights)

                # -- send
                # Get epsilon
                a_epsilon= np.array(epsilon, dtype=np.float64)
                epsilon_win.Lock(0)
                epsilon_win.Put(a_epsilon, target_rank=0)
                epsilon_win.Unlock(0)

                # Get model weight
                model_win.Lock(0)
                model_win.Put(serial_target_weights, target_rank=0)
                model_win.Unlock(0)

                # Get indices
                indices_win.Lock(0)
                indices_win.Put(indices, target_rank=0)
                indices_win.Unlock(0)

                # Get losses
                loss_win.Lock(0)
                loss_win.Put(loss, target_rank=0)
                loss_win.Unlock(0)


            logger.info('Learner time: {}'.format(MPI.Wtime() - start))
            print('Learner time: {}'.format(MPI.Wtime() - start))

        else:
            if mpi_settings.is_actor():
                # Setup logger
                filename_prefix = 'ExaLearner_' + 'Episodes%s_Steps%s_Rank%s_memory_v1' \
                    % (str(workflow.nepisodes), str(workflow.nsteps), str(agent_comm.rank))
                train_file = open(workflow.results_dir + '/' +
                                  filename_prefix + ".log", 'w')
                train_writer = csv.writer(train_file, delimiter=" ")

                # buffers
                buff_target_weights = bytearray(serial_target_weights_size)
                episode_recv = np.ones(1, dtype=np.float64)
                epsilon = np.zeros(1, dtype=np.float64)
                indices = -1 * np.ones(workflow.agent.batch_size, dtype=np.int32)
                loss = np.zeros(workflow.agent.batch_size, dtype=np.float64)
                one = np.ones(1, dtype=np.float64)
                requests = []

            start = MPI.Wtime()
            while True:
                # Reset variables each episode
                workflow.env.seed(0)
                # TODO: optimize some of these variables out for env processes
                current_state = workflow.env.reset()
                total_reward = 0
                steps = 0
                action = 0

                if mpi_settings.is_actor():
                    episode_win.Lock(0)
                    # Atomic Get_accumulate to increment the episode counter
                    episode_win.Get_accumulate(one, episode_recv, target_rank=0, op=MPI.SUM)
                    episode_win.Unlock(0)

                episode = env_comm.bcast(episode_recv, root = 0)[0]
                if episode >= workflow.nepisodes :
                    if mpi_settings.is_actor(): # inform the learner
                        agent_comm.send([-1], dest=0)
                    break # end


                # Steps in an episode
                while steps < workflow.nsteps:
                    logger.debug('ASYNC::run() agent_comm.rank{}; step({} of {})'
                                 .format(agent_comm.rank, steps, (workflow.nsteps - 1)))
                    if mpi_settings.is_actor():

                        # Get epsilon
                        epsilon_win.Lock(0, lock_type=MPI.LOCK_SHARED)
                        epsilon_win.Get(epsilon, target_rank=0)
                        epsilon_win.Unlock(0)

                        # Get model weight
                        model_win.Lock(0, lock_type=MPI.LOCK_SHARED)
                        model_win.Get(buff_target_weights, target_rank=0)
                        model_win.Unlock(0)

                        # Get indices
                        indices_win.Lock(0, lock_type=MPI.LOCK_SHARED)
                        indices_win.Get(indices, target_rank=0)
                        indices_win.Unlock(0)

                        # Get losses
                        loss_win.Lock(0, lock_type=MPI.LOCK_SHARED)
                        loss_win.Get(loss, target_rank=0)
                        loss_win.Unlock(0)


                    if mpi_settings.is_actor():
                        workflow.agent.epsilon = epsilon[0]
                        target_weights = MPI.pickle.loads(buff_target_weights)
                        workflow.agent.set_weights(target_weights)

                        if workflow.action_type == 'fixed':
                            action, policy_type = 0, -11
                        else:
                            action, policy_type = workflow.agent.action(
                                current_state)

                    action = env_comm.bcast(action, root=0)
                    next_state, reward, done, _ = workflow.env.step(action)

                    if mpi_settings.is_actor():
                        total_reward += reward
                        memory = (current_state, action, reward,
                                  next_state, done, total_reward)

                        with open('experience.pkl', 'wb') as f:
                            pickle.dump(memory, f)
                        # batch_data = []
                        workflow.agent.remember(
                            memory[0], memory[1], memory[2], memory[3], memory[4])

                        batch_data = next(workflow.agent.generate_data())
                        logger.info(
                            'Rank[{}] - Generated data: {}'.format(agent_comm.rank, len(batch_data[0])))
                        try:
                            buffer_length = len(workflow.agent.memory)
                        except:
                            buffer_length = workflow.agent.replay_buffer.get_buffer_length()
                        logger.info(
                            'Rank[{}] - # Memories: {}'.format(agent_comm.rank, buffer_length))

                    if steps >= workflow.nsteps - 1:
                        done = True

                    if done :
                        local_episode += 1

                    if mpi_settings.is_actor():
                        # Send batched memories
                        req = agent_comm.isend([0, batch_data, policy_type], dest=0)
                        #agent_comm.send([0, batch_data, policy_type], dest=0)

                        if not req.test()[0]:
                            requests.append(req)

                        if not np.array_equal(indices, (-1 * np.ones(workflow.agent.batch_size, dtype=np.intc))):
                            workflow.agent.set_priorities(indices, loss)
                        logger.info('Rank[%s] - Total Reward:%s' %
                                    (str(agent_comm.rank), str(total_reward)))
                        logger.info(
                            'Rank[%s] - Episode/Step/Status:%s/%s/%s' % (str(agent_comm.rank), str(episode), str(steps), str(done)))

                        train_writer.writerow([time.time(), current_state, action, reward, next_state, total_reward,
                                               done, episode , steps, policy_type, workflow.agent.epsilon])
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
                waiting_start = MPI.Wtime()
                # wait for transfers completition
                for req in requests:
                    req.wait()

        if mpi_settings.is_actor():
            logger.info(f'Agent[{agent_comm.rank}] timing info:\n')
            workflow.agent.print_timers()



        if mpi_settings.is_agent():
            agent_comm.Barrier()
            waiting_end = MPI.Wtime()
            episode_win.Free()
            epsilon_win.Free()
            indices_win.Free()
            loss_win.Free()
            model_win.Free()
            if mpi_settings.is_actor():
                print(" -- async v2 local_episode : ", local_episode, " wainting : ", (waiting_end-waiting_start))
