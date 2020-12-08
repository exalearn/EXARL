import exarl.mpi_settings as mpi_settings
import time
import csv
from mpi4py import MPI
import numpy as np
import exarl as erl
from utils.profile import *
import utils.log as log
import utils.candleDriver as cd
logger = log.setup_logger(__name__, cd.run_params['log_level'])


class ASYNC(erl.ExaWorkflow):
    def __init__(self):
        print('Creating ASYNC learner workflow...')

    @PROFILE
    def run(self, workflow):
        # MPI communicators
        agent_comm = mpi_settings.agent_comm
        env_comm = mpi_settings.env_comm

        # Set target model
        target_weights = None
        if mpi_settings.is_learner():
            workflow.agent.set_learner()
            target_weights = workflow.agent.get_weights()

        # Only agent_comm processes will run this try block
        try:
            # Send and set weights to all other agents
            current_weights = agent_comm.bcast(target_weights, root=0)
            workflow.agent.set_weights(current_weights)
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
                rank0_epsilon = workflow.agent.epsilon
                target_weights = workflow.agent.get_weights()
                episode = worker_episodes[s - 1]
                agent_comm.send(
                    [episode, rank0_epsilon, target_weights], dest=s)

            init_nepisodes = episode
            logger.debug('init_nepisodes:{}'.format(init_nepisodes))

            logger.debug("Continuing ...\n")
            while worker_episodes.max() != -1:
                # print("Running scheduler/workflow episode: {}".format(episode))

                # Receive the rank of the worker ready for more work
                recv_data = agent_comm.recv(source=MPI.ANY_SOURCE)
                whofrom = recv_data[0]
                step = recv_data[1]
                batch = recv_data[2]
                done = recv_data[3]
                logger.debug('step:{}'.format(step))
                logger.debug('done:{}'.format(done))

                if episode_done >= workflow.nepisodes:
                    # Send kill signal
                    agent_comm.send([-1, 0, 0], dest=whofrom)
                else:
                    # Train
                    workflow.agent.train(batch)
                    # TODO: Double check if this is already in the DQN code
                    workflow.agent.target_train()

                    # Send target weights
                    rank0_epsilon = workflow.agent.epsilon
                    logger.debug('rank0_epsilon:{}'.format(rank0_epsilon))

                    target_weights = workflow.agent.get_weights()

                    # Increment episode when starting
                    if step == 0:
                        episode += 1
                        logger.debug('if episode:{}'.format(episode))

                    # Increment the number of completed episodes
                    if done:
                        episode_done += 1
                        latest_episode = worker_episodes.max()
                        if latest_episode < workflow.nepisodes:
                            worker_episodes[whofrom - 1] = latest_episode + 1
                        else:
                            worker_episodes[whofrom - 1] = -1
                        logger.debug('episode_done:{}'.format(episode_done))

                    agent_comm.send([worker_episodes[whofrom - 1],
                                 rank0_epsilon, target_weights], dest=whofrom)

            logger.info("Finishing up ...\n")
            logger.info('Learner time: {}'.format(MPI.Wtime() - start))

        elif mpi_settings.is_actor():
            # Setup logger
            filename_prefix = 'ExaLearner_' + 'Episodes%s_Steps%s_Rank%s_memory_v1' \
                % (str(workflow.nepisodes), str(workflow.nsteps), str(agent_comm.rank))
            train_file = open(workflow.results_dir + '/' +
                                filename_prefix + ".log", 'w')
            train_writer = csv.writer(train_file, delimiter=" ")

            start = MPI.Wtime()
            while episode != -1:
                # Add start jitter to stagger the jobs [ 1-50 milliseconds]
                # time.sleep(randint(0, 50) / 1000)
                # Reset variables each episode
                workflow.env.seed(0)
                # TODO: optimize some of these variables out for env processes
                current_state = workflow.env.reset()
                total_reward = 0
                steps = 0
                action = 0

                # Steps in an episode
                while steps < workflow.nsteps:
                    # Receive target weights
                    recv_data = agent_comm.recv(source=0)
                    episode = recv_data[0]

                    if episode == -1:
                        logger.info('Rank[%s] - Episode/Step:%s/%s' %
                                    (str(agent_comm.rank), str(episode), str(steps)))
                        break

                    workflow.agent.epsilon = recv_data[1]
                    workflow.agent.set_weights(recv_data[2])

                    action, policy_type = workflow.agent.action(current_state)
                    if workflow.action_type == 'fixed':
                        action, policy_type = 0, -11
                            
                    next_state, reward, done, _ = workflow.env.step(action)
                    
                    total_reward += reward
                    memory = (current_state, action, reward,
                            next_state, done, total_reward)

                    # batch_data = []
                    workflow.agent.remember(
                        memory[0], memory[1], memory[2], memory[3], memory[4])

                    batch_data = next(workflow.agent.generate_data())
                    logger.info(
                        'Rank[{}] - Generated data: {}'.format(agent_comm.rank, len(batch_data[0])))
                    logger.info(
                        'Rank[{}] - Memories: {}'.format(agent_comm.rank, len(workflow.agent.memory)))

                    if steps >= workflow.nsteps - 1:
                        done = True

                    # Send batched memories
                    agent_comm.send(
                        [agent_comm.rank, steps, batch_data, done], dest=0)

                    logger.info('Rank[%s] - Total Reward:%s' %
                                (str(agent_comm.rank), str(total_reward)))
                    logger.info(
                        'Rank[%s] - Episode/Step/Status:%s/%s/%s' % (str(agent_comm.rank), str(episode), str(steps), str(done)))

                    train_writer.writerow([time.time(), current_state, action, reward, next_state, total_reward,
                                        done, episode, steps, policy_type, workflow.agent.epsilon])
                    train_file.flush()

                    # Update state and step
                    current_state = next_state
                    steps += 1

                    # Broadcast done
                    # done = env_comm.bcast(done, root=0)
                    # # Break for loop if done
                    # if done:
                    #     break
                logger.info('Worker time = {}'.format(MPI.Wtime() - start))

            train_file.close()

            logger.info(f'Agent[{agent_comm.rank}] timing info:\n')
            workflow.agent.print_timers()
