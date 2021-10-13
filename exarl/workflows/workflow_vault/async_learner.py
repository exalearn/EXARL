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
import exarl as erl
from exarl.utils.introspect import ib
from exarl.utils.profile import *
from exarl.utils import log
import exarl.utils.candleDriver as cd
from exarl.base.comm_base import ExaComm

logger = log.setup_logger(__name__, cd.run_params['log_level'])

class ASYNC(erl.ExaWorkflow):
    def __init__(self):
        print('Creating ASYNC learner workflow...')
        priority_scale = cd.lookup_params('priority_scale')
        self.use_priority_replay = (priority_scale is not None and priority_scale > 0)

    @PROFILE
    def run(self, workflow):
        # MPI communicators
        agent_comm = ExaComm.agent_comm
        env_comm = ExaComm.env_comm

        target_weights = None

        # Variables for all
        episode = 0
        episode_done = 0
        episode_interim = 0

        # Round-Robin Scheduler
        if ExaComm.is_learner():
            start = agent_comm.time()
            worker_episodes = np.arange(1, agent_comm.size)
            logger.debug("worker_episodes:{}".format(worker_episodes))

            logger.info("Initializing ...\n")
            for s in range(1, agent_comm.size):
                # Send target weights
                indices, loss = None, None
                rank0_epsilon = workflow.agent.epsilon
                target_weights = workflow.agent.get_weights()
                episode = worker_episodes[s - 1]
                agent_comm.send(
                    [episode, rank0_epsilon, target_weights, indices, loss], dest=s)

            init_nepisodes = episode
            logger.debug("init_nepisodes:{}".format(init_nepisodes))

            logger.debug("Continuing ...\n")
            while episode_done < workflow.nepisodes:
                # Receive the rank of the worker ready for more work
                recv_data = agent_comm.recv(None)
                ib.update("Async_Learner_Get_Data", 1)

                whofrom = recv_data[0]
                step = recv_data[1]
                batch = recv_data[2]
                policy_type = recv_data[3]
                done = recv_data[4]
                logger.debug('step:{}'.format(step))
                logger.debug('done:{}'.format(done))
                # Train
                train_return = workflow.agent.train(batch)
                ib.update("Async_Learner_Train", 1)
                # if train_return is not None:
                if self.use_priority_replay and train_return is not None:
                    if train_return[0][0] != -1:
                        indices, loss = train_return
                # TODO: Double check if this is already in the DQN code
                workflow.agent.target_train()
                ib.update("Async_Learner_Target_Train", 1)

                if policy_type == 0:
                    workflow.agent.epsilon_adj()
                epsilon = workflow.agent.epsilon

                # Send target weights
                logger.debug('rank0_epsilon:{}'.format(epsilon))
                # Increment episode when starting
                if step == 0:
                    episode += 1
                    logger.debug("if episode:{}".format(episode))

                # Increment the number of completed episodes
                if done:
                    episode_done += 1
                    latest_episode = worker_episodes.max()
                    # Updated episode = latest_episode + 1
                    worker_episodes[whofrom - 1] = latest_episode + 1
                    logger.debug("episode_done:{}".format(episode_done))
                    ib.update("Async_Learner_Episode", 1)

                # Send target weights
                logger.debug('rank0_epsilon:{}'.format(epsilon))
                target_weights = workflow.agent.get_weights()
                agent_comm.send([worker_episodes[whofrom - 1], epsilon, target_weights, indices, loss], whofrom)

            filename_prefix = 'ExaLearner_Episodes%s_Steps%s_Rank%s_memory_v1' \
                % (str(workflow.nepisodes), str(workflow.nsteps), str(agent_comm.rank))
            workflow.agent.save(workflow.results_dir + '/' + filename_prefix + '.h5')
            logger.info("Finishing up ...\n")
            episode = -1
            for s in range(1, agent_comm.size):
                recv_data = agent_comm.recv(None)
                whofrom = recv_data[0]
                step = recv_data[1]
                batch = recv_data[2]
                epsilon = recv_data[3]
                done = recv_data[4]
                logger.debug('step:{}'.format(step))
                logger.debug('done:{}'.format(done))

                # Train
                train_return = workflow.agent.train(batch)
                if train_return is not None:
                    indices, loss = train_return
                workflow.agent.target_train()
                # Save weights
                if ExaComm.learner_comm.rank == 0:
                    target_weights = workflow.agent.get_weights()
                    workflow.agent.save(workflow.results_dir + '/target_weights.pkl')
                agent_comm.send([episode, 0, 0, indices, loss], s)

            logger.info("Learner time: {}".format(agent_comm.time() - start))

        else:
            if ExaComm.env_comm.rank == 0:
                # Setup logger
                filename_prefix = 'ExaLearner_Episodes%s_Steps%s_Rank%s_memory_v1' \
                    % (str(workflow.nepisodes), str(workflow.nsteps), str(agent_comm.rank))
                train_file = open(workflow.results_dir + '/' +
                                  filename_prefix + ".log", 'w')
                train_writer = csv.writer(train_file, delimiter=" ")

            start = env_comm.time()
            while episode != -1:
                # Reset variables each episode
                # workflow.env.seed(0)
                # TODO: optimize some of these variables out for env processes
                current_state = workflow.env.reset()
                total_reward = 0
                steps = 0
                action = 0
                episode_reward_list = []

                # Steps in an episode
                while steps < workflow.nsteps:
                    logger.debug('ASYNC::run() agent_comm.rank{}; step({} of {})'
                                 .format(agent_comm.rank, steps, (workflow.nsteps - 1)))
                    if ExaComm.env_comm.rank == 0:
                        # Receive target weights
                        recv_data = agent_comm.recv(None, source=0)
                        # Update episode while beginning a new one i.e. step = 0
                        if steps == 0:
                            episode = recv_data[0]
                        # This variable is used for kill check
                        episode_interim = recv_data[0]

                    # Broadcast episode within env_comm
                    episode_interim = env_comm.bcast(episode_interim, 0)

                    if episode_interim == -1:
                        episode = -1
                        if ExaComm.env_comm.rank == 0:
                            logger.info(
                                "Rank[%s] - Episode/Step:%s/%s"
                                % (str(agent_comm.rank), str(episode), str(steps))
                            )
                        break

                    send_data = False
                    done = False
                    while send_data == False and done == False:
                        if ExaComm.env_comm.rank == 0:
                            workflow.agent.epsilon = recv_data[1]
                            workflow.agent.set_weights(recv_data[2])

                            action, policy_type = workflow.agent.action(current_state)
                            ib.update("Async_Env_Inference", 1)

                            if workflow.action_type == "fixed":
                                action, policy_type = 0, -11

                        # Broadcast episode count to all procs in env_comm
                        action = env_comm.bcast(action, root)

                        ib.startTrace("step", 0)
                        next_state, reward, done, _ = workflow.env.step(action)
                        ib.stopTrace()
                        ib.update("Async_Env_Step", 1)

                        if ExaComm.env_comm.rank == 0:
                            total_reward += reward
                            # memory = (
                            #     current_state,
                            #     action,
                            #     reward,
                            #     next_state,
                            #     done,
                            #     total_reward,
                            # )
                            # workflow.agent.remember(memory[0], memory[1], memory[2], memory[3], memory[4])
                            workflow.agent.remember(current_state, action, reward, next_state, done)

                            batch_data = next(workflow.agent.generate_data())
                            ib.update("Async_Env_Generate_Data", 1)

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

                        if ExaComm.env_comm.rank == 0:
                            # Send batched memories
                            if workflow.agent.has_data():
                                send_data = True
                                agent_comm.send([agent_comm.rank, steps, batch_data, policy_type, done], 0)
                            indices, loss = recv_data[3:5]
                            if indices is not None:
                                workflow.agent.set_priorities(indices, loss)
                            logger.info('Rank[%s] - Total Reward:%s' %
                                        (str(agent_comm.rank), str(total_reward)))
                            logger.info(
                                'Rank[%s] - Episode/Step/Status:%s/%s/%s' % (str(agent_comm.rank), str(episode), str(steps), str(done)))

                            # TODO: make this configurable so we don't always suffer IO
                            train_writer.writerow([time.time(), current_state, action, reward, next_state, total_reward,
                                                   done, episode, steps, policy_type, workflow.agent.epsilon])
                            train_file.flush()

                        # Update state and step
                        current_state = next_state
                        steps += 1

                        # Broadcast done
                        done = env_comm.bcast(done, 0)
                    # Break loop if done
                    if done:
                        break
                episode_reward_list.append(total_reward)
                # Mean of last 40 episodes
                average_reward = np.mean(episode_reward_list[-40:])
                print("Episode * {} * Avg Reward is ==> {}".format(episode, average_reward))
            ib.update("Async_Env_Episode", 1)
            logger.info("Worker time = {}".format(env_comm.time() - start))
            if ExaComm.is_actor():
                train_file.close()
