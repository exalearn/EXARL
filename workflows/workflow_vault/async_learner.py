import exarl.mpi_settings as mpi_settings
import time
import csv
from mpi4py import MPI
import numpy as np
import exarl as erl

import utils.log as log
from utils.candleDriver import initialize_parameters
run_params = initialize_parameters()
logger = log.setup_logger('RL-Logger', run_params['log_level'])


class ASYNC(erl.ExaWorkflow):
    def __init__(self):
        print('Class ASYNC learner')

    def run(self, learner):

        # MPI communicators
        agent_comm = mpi_settings.agent_comm
        env_comm = mpi_settings.env_comm

        # Set target model
        target_weights = None
        if mpi_settings.is_learner():
            learner.agent.set_learner()
            target_weights = learner.agent.get_weights()

        # Only agent_comm processes will run this try block
        try:
            # Send and set weights to all other agents
            current_weights = agent_comm.bcast(target_weights, root=0)
            learner.agent.set_weights(current_weights)
        except:
            logger.debug('Does not contain an agent')

        # Variables for all
        episode = 0
        episode_done = 0
        episode_interim = 0

        # Round-Robin Scheduler
        if mpi_settings.is_learner():
            start = MPI.Wtime()
            # worker_episodes = np.linspace(0, agent_comm.size - 2, agent_comm.size - 1)
            worker_episodes = np.arange(1, agent_comm.size)
            logger.debug('worker_episodes:{}'.format(worker_episodes))

            logger.info("Initializing ...\n")
            for s in range(1, agent_comm.size):
                # Send target weights
                rank0_epsilon = learner.agent.epsilon
                target_weights = learner.agent.get_weights()
                episode = worker_episodes[s - 1]
                print('send inside the initialize')
                agent_comm.send(
                    [episode, rank0_epsilon, target_weights], dest=s)

            init_nepisodes = episode
            logger.debug('init_nepisodes:{}'.format(init_nepisodes))

            logger.debug("Continuing ...\n")
            while episode_done < learner.nepisodes:
                # print("Running scheduler/learner episode: {}".format(episode))

                # Receive the rank of the worker ready for more work
                recv_data = agent_comm.recv(source=MPI.ANY_SOURCE)

                whofrom = recv_data[0]
                step = recv_data[1]
                batch = recv_data[2]
                done = recv_data[3]
                logger.debug('step:{}'.format(step))
                logger.debug('done:{}'.format(done))
                # Train
                learner.agent.train(batch)
                # TODO: Double check if this is already in the DQN code
                learner.agent.target_train()

                # Send target weights
                rank0_epsilon = learner.agent.epsilon
                logger.debug('rank0_epsilon:{}'.format(rank0_epsilon))

                target_weights = learner.agent.get_weights()

                # Increment episode when starting
                if step == 0:
                    episode += 1
                    logger.debug('if episode:{}'.format(episode))

                # Increment the number of completed episodes
                if done:
                    episode_done += 1
                    latest_episode = worker_episodes.max()
                    worker_episodes[whofrom - 1] = latest_episode + 1
                    logger.debug('episode_done:{}'.format(episode_done))

                agent_comm.send([worker_episodes[whofrom - 1],
                                 rank0_epsilon, target_weights], dest=whofrom)

            logger.info("Finishing up ...\n")
            episode = -1
            for s in range(1, agent_comm.size):
                agent_comm.send([episode, 0, 0], dest=s)

            logger.info('Learner time: {}'.format(MPI.Wtime() - start))

        else:
            if mpi_settings.is_actor():
                # Setup logger
                filename_prefix = 'ExaLearner_' + 'Episodes%s_Steps%s_Rank%s_memory_v1' \
                    % (str(learner.nepisodes), str(learner.nsteps), str(agent_comm.rank))
                train_file = open(learner.results_dir + '/' +
                                  filename_prefix + ".log", 'w')
                train_writer = csv.writer(train_file, delimiter=" ")

            start = MPI.Wtime()
            while episode != -1:
                # Add start jitter to stagger the jobs [ 1-50 milliseconds]
                # time.sleep(randint(0, 50) / 1000)
                # Reset variables each episode
                learner.env.seed(0)
                # TODO: optimize some of these variables out for env processes
                current_state = learner.env.reset()
                total_reward = 0
                steps = 0
                action = 0

                # Steps in an episode
                while steps < learner.nsteps:
                    if mpi_settings.is_actor():
                        # Receive target weights
                        recv_data = agent_comm.recv(source=0)
                        # Update episode while beginning a new one i.e. step = 0
                        if steps == 0:
                            episode = recv_data[0]
                        # This variable is used for kill check
                        episode_interim = recv_data[0]

                    # Broadcast episode within env_comm
                    episode_interim = env_comm.bcast(episode_interim, root=0)

                    if episode_interim == -1:
                        episode = -1
                        if mpi_settings.is_actor():
                            logger.info('Rank[%s] - Episode/Step:%s/%s' %
                                        (str(agent_comm.rank), str(episode), str(steps)))
                        break

                    if mpi_settings.is_actor():
                        learner.agent.epsilon = recv_data[1]
                        learner.agent.set_weights(recv_data[2])

                        if learner.action_type == 'fixed':
                            action, policy_type = 0, -11
                        else:
                            action, policy_type = learner.agent.action(
                                current_state)

                    next_state, reward, done, _ = learner.env.step(action)

                    if mpi_settings.is_actor():
                        total_reward += reward
                        memory = (current_state, action, reward,
                                  next_state, done, total_reward)

                        # batch_data = []
                        learner.agent.remember(
                            memory[0], memory[1], memory[2], memory[3], memory[4])

                        batch_data = next(learner.agent.generate_data())
                        logger.info(
                            'Rank[{}] - Generated data: {}'.format(agent_comm.rank, len(batch_data[0])))
                        logger.info(
                            'Rank[{}] - Memories: {}'.format(agent_comm.rank, len(learner.agent.memory)))

                    if steps >= learner.nsteps - 1:
                        done = True

                    if mpi_settings.is_actor():
                        # Send batched memories
                        agent_comm.send(
                            [agent_comm.rank, steps, batch_data, done], dest=0)

                        logger.info('Rank[%s] - Total Reward:%s' %
                                    (str(agent_comm.rank), str(total_reward)))
                        logger.info(
                            'Rank[%s] - Episode/Step/Status:%s/%s/%s' % (str(agent_comm.rank), str(episode), str(steps), str(done)))

                        train_writer.writerow([time.time(), current_state, action, reward, next_state, total_reward,
                                               done, episode, steps, policy_type, learner.agent.epsilon])
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

        if mpi_settings.is_actor():
            logger.info(f'Agent[{agent_comm.rank}] timing info:\n')
            learner.agent.print_timers()
