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


class SEED_A2C(erl.ExaWorkflow):
    def __init__(self):
        print('Creating SEED A2C learner workflow...')

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

        # Set target model
        target_weights = None
        if mpi_settings.is_learner():
            workflow.agent.set_learner()
            target_weights = workflow.agent.get_weights()
            current_weights = learner_comm.bcast(target_weights, root=0)
            workflow.agent.set_weights(current_weights)

        # Variables for all
        episode = 0
        episode_done = 0
        local_episode = 0
        generate_data_time = 0
        # Round-Robin Scheduler
        if mpi_settings.is_learner():

            status = MPI.Status()
            start = MPI.Wtime()

            logger.debug("Continuing ...\n")
            while episode_done < workflow.nepisodes:
                # Receive the rank of the worker ready for more work
                recv_data = agent_comm.recv(source=MPI.ANY_SOURCE, status=status)

                if recv_data[0] == 0: # a request for an action
                    state = recv_data[1]
                    if workflow.action_type == 'fixed':
                        action, policy_type = 0, -11
                    else:
                        action, policy_type = workflow.agent.action(state)
                    # send the action to the actor
                    agent_comm.send([action, policy_type, workflow.agent.epsilon], dest=status.source)

                elif recv_data[0] == 1 : # recv observation
                    # construct batch
                    batch = recv_data[1]
                    policy_type = recv_data[2]
                    done = recv_data[3]

                    # Train
                    train_return = workflow.agent.train(batch)
                    #if train_return is not None:
                        #if not np.array_equal(train_return[0], (-1 * np.ones(workflow.agent.batch_size))):
                            #indices, loss = train_return

                    workflow.agent.target_train()
                    if policy_type == 0:
                        workflow.agent.epsilon_adj()

                    logger.debug('rank0_epsilon:{}'.format(workflow.agent.epsilon))
                    # dump target weights
                    target_weights = workflow.agent.get_weights()
                    with open('target_weights.pkl', 'wb') as f:
                        pickle.dump(target_weights, f)

                    # increment episode_done number
                    if done :
                        episode_done+=1
                        logger.debug('episode_done:{}'.format(episode_done))



            logger.info('Learner time: {}'.format(MPI.Wtime() - start))
            print("[Learner] total time : {} , generate data time : {}".format(MPI.Wtime() - start,generate_data_time))

        else:
            if mpi_settings.is_actor():
                # Setup logger
                filename_prefix = 'ExaLearner_' + 'Episodes%s_Steps%s_Rank%s_memory_v1' \
                    % (str(workflow.nepisodes), str(workflow.nsteps), str(agent_comm.rank))
                train_file = open(workflow.results_dir + '/' +
                                  filename_prefix + ".log", 'w')
                train_writer = csv.writer(train_file, delimiter=" ")

            # buffers
            episode_recv = np.ones(1, dtype=np.float64)
            one = np.ones(1, dtype=np.float64)

            start = MPI.Wtime()
            while True:
                # Reset variables each episode
                workflow.env.seed(0)
                # TODO: optimize some of these variables out for env processes
                current_state = workflow.env.reset()
                total_reward = 0
                steps = 0
                action = 0
                epsilon = 0

                if mpi_settings.is_actor():
                    episode_win.Lock(0)
                    # Atomic Get_accumulate to increment the episode counter
                    episode_win.Get_accumulate(one, episode_recv, target_rank=0, op=MPI.SUM)
                    episode_win.Unlock(0)
                    episode=episode_recv[0]

                episode = env_comm.bcast(episode, root=0)
                # end loop
                if episode >= workflow.nepisodes :
                    break

                # Steps in an episode
                while steps < workflow.nsteps:
                    logger.debug('ASYNC::run() agent_comm.rank{}; step({} of {})'
                                 .format(agent_comm.rank, steps, (workflow.nsteps - 1)))

                    if mpi_settings.is_actor():
                        agent_comm.send([0,current_state],dest=0) # ask for action
                        recv_data = agent_comm.recv(source=0)
                        action = recv_data[0]
                        policy_type = recv_data[1]
                        epsilon = recv_data[2]

                    action = env_comm.bcast(action, root=0)
                    next_state, reward, done, _ = workflow.env.step(action)

                    if mpi_settings.is_actor():
                        total_reward += reward
                        memory = (current_state, action, reward, next_state, done, total_reward)

                        with open('experience.pkl', 'wb') as f:
                            pickle.dump(memory, f)

                        # memory
                        workflow.agent.remember(memory[0], memory[1], memory[2], memory[3], memory[4])

                    if steps >= workflow.nsteps - 1:
                        done = True

                    if done :
                        local_episode += 1

                    if mpi_settings.is_actor():
                        # Send observation
                        if done :
                            batch_data = workflow.agent.generate_data()
                            agent_comm.send([1, batch_data, done, policy_type], dest=0)

                        train_writer.writerow([time.time(), current_state, action, reward, next_state, total_reward,
                                               done, episode, steps, policy_type, epsilon])
                        train_file.flush()

                    # Update state and step
                    current_state = next_state
                    steps += 1

                    # Broadcast done
                    done = env_comm.bcast(done, root=0)
                    # Break for loop if done
                    if done:
                        if mpi_settings.is_actor():
                            workflow.agent.reset_lists()
                        break

            logger.info('Worker time = {}'.format(MPI.Wtime() - start))
            if mpi_settings.is_actor():
                train_file.close()

        if mpi_settings.is_agent():
            agent_comm.Barrier()
            episode_win.Free()

        if mpi_settings.is_actor():
            logger.info(f'Agent[{agent_comm.rank}] timing info:\n')
            #workflow.agent.print_timers()
            print(" -- local_episode : ", local_episode)
