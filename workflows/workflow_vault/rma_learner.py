import exarl.mpi_settings as mpi_settings
# import time
# import csv
from mpi4py import MPI
import numpy as np
import exarl as erl
from utils.profile import *
import utils.log as log
import utils.candleDriver as cd

logger = log.setup_logger(__name__, cd.run_params['log_level'])

class RMA_ASYNC(erl.ExaWorkflow):
    def __init__(self):
        print('Creating RMA async workflow...')

    def run(self, workflow):
        # MPI communicators
        agent_comm = mpi_settings.agent_comm
        env_comm = mpi_settings.env_comm

        # Workflow level variables
        episode_count = np.zeros(1)

        if mpi_settings.is_learner():
            workflow.agent.set_learner()

        if mpi_settings.is_agent():
            target_weights = workflow.agent.get_weights()
            serial_target_weights = MPI.pickle.dumps(target_weights)
            episode_size = 0
            target_weights_size = 0
            if mpi_settings.is_actor():
                target_weights_size = len(serial_target_weights)
                episode_size = MPI.INT64_T.Get_size()
            model_win = MPI.Win.Allocate(target_weights_size, 1, comm=agent_comm)
            episode_win = MPI.Win.Allocate(episode_size, 1, comm=agent_comm)

        if mpi_settings.is_learner():
            # Send all target weight
            for s in range(1, agent_comm.size):
                model_win.Lock(s)
                model_win.Put(serial_target_weights, target_rank=s)
                model_win.Unlock(s)
            # Define the episode counter window

        # Get data window -- WILL NOT WORK --
        # Save on each actor?
        agent_batch =  next(workflow.agent.generate_data())
        serial_agent_batch = (MPI.pickle.dumps(agent_batch))
        nserial_agent_batch = len(serial_agent_batch)
        data_win = MPI.Win.Allocate(nserial_agent_batch, 1, comm=agent_comm)

        print('Init done ...')
        agent_comm.Barrier()

        # Learner
        if mpi_settings.is_learner():
            while 1:
                # 0) Check if exit boolean
                episode_win.Lock(0)
                episode_win.Get(episode_count, target_rank=0, target=None)
                # print('Rank[{}] - working on episode: {}'.format(agent_comm.rank,episode_count))
                episode_win.Unlock(0)
                if episode_count < workflow.nepisodes:
                    print('Learner exit on episode: {}'.format(agent_comm.rank, episode_count))
                    break

                # Loop over all actor data, train, and update model
                for s in range(1, agent_comm.size):
                    # 1) Get data
                    data_win.Lock(s)
                    data_win.Get(serial_agent_batch, target_rank=s, target=None)
                    data_win.Unlock(s)
                    agent_data = MPI.pickle.loads(serial_agent_batch)
                    # 2) Train & Target train
                    workflow.agent.train(agent_data)
                    # TODO: Double check if this is already in the DQN code
                    workflow.agent.target_train()
                    # 3) Share new model weights
                    target_weights = workflow.agent.get_weights()
                    serial_target_weights = MPI.pickle.dumps(target_weights)
                    # TODO: loop over all actors (?)
                    model_win.Lock(s)
                    model_win.Put(serial_target_weights, target_rank=s)
                    model_win.Unlock(s)

            print('Learner exit on episode: {}'.format(agent_comm.rank, episode_count))
        else:

            while episode_count < workflow.nepisodes:
                workflow.env.seed(0)
                current_state = workflow.env.reset()
                total_rewards = 0
                steps = 0
                action = 0

                while steps < workflow.nsteps:
                    # 0) Check episode counter
                    episode_win.Lock(0)
                    episode_win.Get(episode_count, target_rank=0, target=None)
                    episode_win.Unlock(0)
                    if episode_count >= workflow.nepisodes:
                        break

                    # 1) Update model weight
                    # TODO: weights are updated each step -- REVIEW --
                    buff = bytearray(target_weights_size)
                    model_win.Lock(agent_comm.rank)
                    model_win.Get(buff,target=0, target_rank=agent_comm.rank)
                    model_win.Unlock(agent_comm.rank)
                    target_weights = MPI.pickle.loads(buff)
                    workflow.agent.set_weights(target_weights)

                    # 2) Inference action
                    if mpi_settings.is_actor():
                        if workflow.action_type == 'fixed':
                            action, policy_type = 0, -11
                        else:
                            action, policy_type = workflow.agent.action(current_state)

                    # 3) Env step
                    next_state, reward, done, _ = workflow.env.step(action)

                    steps += 1
                    if steps >= workflow.nsteps:
                        done = True

                    # 4) If done then update the episode counter and exit boolean
                    # TODO: Verify this with a toy example
                    if done:
                        episode_win.Lock(0)
                        episode_win.Get(episode_count, target_rank=0, target=None)
                        episode_count += 1
                        print('Rank[{}] - working on episode: {}'.format(agent_comm.rank, episode_count))
                        episode_win.Put(episode_count, target_rank=0)
                        episode_win.Unlock(0)
                #
                # # 4) Generate new training data
                # if mpi_settings.is_actor():
                #     total_reward += reward
                #     memory = (current_state, action, reward,
                #           next_state, done, total_reward)
                #
                #     # batch_data = []
                #     workflow.agent.remember(
                #         memory[0], memory[1], memory[2], memory[3], memory[4])
                #
                #     batch_data = next(workflow.agent.generate_data())
                #
                #     serial_batch_data = MPI.pickle.dumps(batch_data)
                #     # TODO: Consider MPI_Accumulate with replace
                #     data_win.Lock(0)
                #     data_win.Put(serial_batch_data, 0, target = (agent_comm.rank-1))
                #     data_win.Unlock(0)

        #     start = MPI.Wtime()
        #     # worker_episodes = np.linspace(0, agent_comm.size - 2, agent_comm.size - 1)
        #     worker_episodes = np.arange(1, agent_comm.size)
        #     logger.debug('worker_episodes:{}'.format(worker_episodes))
        #
        #     logger.info("Initializing ...\n")
        #     for s in range(1, agent_comm.size):
        #         # Send target weights
        #         rank0_epsilon = workflow.agent.epsilon
        #         target_weights = workflow.agent.get_weights()
        #         episode = worker_episodes[s - 1]
        #         # TODO: Change to put
        #         agent_comm.send(
        #             [episode, rank0_epsilon, target_weights], dest=s)
        #
        #     init_nepisodes = episode
        #     logger.debug('init_nepisodes:{}'.format(init_nepisodes))
        #
        #     logger.debug("Continuing ...\n")
        #     while episode_done < workflow.nepisodes:
        #         # print("Running scheduler/workflow episode: {}".format(episode))
        #
        #         # Receive the rank of the worker ready for more work
        #         recv_data = agent_comm.recv(source=MPI.ANY_SOURCE)
        #
        #         whofrom = recv_data[0]
        #         step = recv_data[1]
        #         batch = recv_data[2]
        #         done = recv_data[3]
        #         logger.debug('step:{}'.format(step))
        #         logger.debug('done:{}'.format(done))
        #         # Train
        #         workflow.agent.train(batch)
        #         # TODO: Double check if this is already in the DQN code
        #         workflow.agent.target_train()
        #
        #         # Send target weights
        #         rank0_epsilon = workflow.agent.epsilon
        #         logger.debug('rank0_epsilon:{}'.format(rank0_epsilon))
        #
        #         target_weights = workflow.agent.get_weights()
        #
        #         # Increment episode when starting
        #         if step == 0:
        #             episode += 1
        #             logger.debug('if episode:{}'.format(episode))
        #
        #         # Increment the number of completed episodes
        #         if done:
        #             episode_done += 1
        #             latest_episode = worker_episodes.max()
        #             worker_episodes[whofrom - 1] = latest_episode + 1
        #             logger.debug('episode_done:{}'.format(episode_done))
        #
        #         agent_comm.send([worker_episodes[whofrom - 1],
        #                          rank0_epsilon, target_weights], dest=whofrom)
        #
        #     logger.info("Finishing up ...\n")
        #     episode = -1
        #     for s in range(1, agent_comm.size):
        #         recv_data = agent_comm.recv(source=MPI.ANY_SOURCE)
        #         whofrom = recv_data[0]
        #         step = recv_data[1]
        #         batch = recv_data[2]
        #         done = recv_data[3]
        #         logger.debug('step:{}'.format(step))
        #         logger.debug('done:{}'.format(done))
        #         # Train
        #         workflow.agent.train(batch)
        #         # TODO: Double check if this is already in the DQN code
        #         workflow.agent.target_train()
        #         agent_comm.send([episode, 0, 0], dest=s)
        #
        #     logger.info('workflow time: {}'.format(MPI.Wtime() - start))
        #
        # else:
        #     if mpi_settings.is_actor():
        #         # Setup logger
        #         filename_prefix = 'Exaworkflow_' + 'Episodes%s_Steps%s_Rank%s_memory_v1' \
        #             % (str(workflow.nepisodes), str(workflow.nsteps), str(agent_comm.rank))
        #         train_file = open(workflow.results_dir + '/' +
        #                           filename_prefix + ".log", 'w')
        #         train_writer = csv.writer(train_file, delimiter=" ")
        #
        #     start = MPI.Wtime()
        #     while episode != -1:
        #         # Add start jitter to stagger the jobs [ 1-50 milliseconds]
        #         # time.sleep(randint(0, 50) / 1000)
        #         # Reset variables each episode
        #         workflow.env.seed(0)
        #         # TODO: optimize some of these variables out for env processes
        #         current_state = workflow.env.reset()
        #         total_reward = 0
        #         steps = 0
        #         action = 0
        #
        #         # Steps in an episode
        #         while steps < workflow.nsteps:
        #             logger.debug('ASYNC::run() agent_comm.rank{}; step({} of {})'
        #                          .format(agent_comm.rank, steps, (workflow.nsteps - 1)))
        #             if mpi_settings.is_actor():
        #                 # Receive target weights
        #                 recv_data = agent_comm.recv(source=0)
        #                 # Update episode while beginning a new one i.e. step = 0
        #                 if steps == 0:
        #                     episode = recv_data[0]
        #                 # This variable is used for kill check
        #                 episode_interim = recv_data[0]
        #
        #             # Broadcast episode within env_comm
        #             episode_interim = env_comm.bcast(episode_interim, root=0)
        #
        #             if episode_interim == -1:
        #                 episode = -1
        #                 if mpi_settings.is_actor():
        #                     logger.info('Rank[%s] - Episode/Step:%s/%s' %
        #                                 (str(agent_comm.rank), str(episode), str(steps)))
        #                 break
        #
        #             if mpi_settings.is_actor():
        #                 workflow.agent.epsilon = recv_data[1]
        #                 workflow.agent.set_weights(recv_data[2])
        #
        #                 if workflow.action_type == 'fixed':
        #                     action, policy_type = 0, -11
        #                 else:
        #                     action, policy_type = workflow.agent.action(
        #                         current_state)
        #
        #             next_state, reward, done, _ = workflow.env.step(action)
        #
        #             if mpi_settings.is_actor():
        #                 total_reward += reward
        #                 memory = (current_state, action, reward,
        #                           next_state, done, total_reward)
        #
        #                 # batch_data = []
        #                 workflow.agent.remember(
        #                     memory[0], memory[1], memory[2], memory[3], memory[4])
        #
        #                 batch_data = next(workflow.agent.generate_data())
        #                 logger.info(
        #                     'Rank[{}] - Generated data: {}'.format(agent_comm.rank, len(batch_data[0])))
        #                 logger.info(
        #                     'Rank[{}] - Memories: {}'.format(agent_comm.rank, len(workflow.agent.memory)))
        #
        #             if steps >= workflow.nsteps - 1:
        #                 done = True
        #
        #             if mpi_settings.is_actor():
        #                 # Send batched memories
        #                 agent_comm.send(
        #                     [agent_comm.rank, steps, batch_data, done], dest=0)
        #
        #                 logger.info('Rank[%s] - Total Reward:%s' %
        #                             (str(agent_comm.rank), str(total_reward)))
        #                 logger.info(
        #                     'Rank[%s] - Episode/Step/Status:%s/%s/%s' % (str(agent_comm.rank), str(episode), str(steps), str(done)))
        #
        #                 train_writer.writerow([time.time(), current_state, action, reward, next_state, total_reward,
        #                                        done, episode, steps, policy_type, workflow.agent.epsilon])
        #                 train_file.flush()
        #
        #             # Update state and step
        #             current_state = next_state
        #             steps += 1
        #
        #             # Broadcast done
        #             done = env_comm.bcast(done, root=0)
        #             # Break for loop if done
        #             if done:
        #                 break
        #     logger.info('Worker time = {}'.format(MPI.Wtime() - start))
        #     if mpi_settings.is_actor():
        #         train_file.close()
        #
        # if mpi_settings.is_actor():
        #     logger.info(f'Agent[{agent_comm.rank}] timing info:\n')
        #     workflow.agent.print_timers()
