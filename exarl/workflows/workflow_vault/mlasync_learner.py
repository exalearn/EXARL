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


class ML_ASYNC(erl.ExaWorkflow):
    def __init__(self):
        print('Creating ML_ASYNC learner workflow...')

    @PROFILE
    def run(self, workflow):

        # MPI communicators
        agent_comm = mpi_settings.agent_comm
        env_comm = mpi_settings.env_comm
        learner_comm = mpi_settings.learner_comm

        actor_procs = 0

        # number of learner procs
        learner_procs = int(cd.run_params['learner_procs'])
        if mpi_settings.is_agent():
            actor_procs = agent_comm.size - learner_procs

        actor_procs = env_comm.bcast(actor_procs, root=0)

        # Set target model
        target_weights = None
        # actr_recv_counter = 0
        # actr_send_counter = 0
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

            indices, loss = None, None
            rank0_epsilon = None
            target_weights = None
            data_buff = None
            counter = 0
            index = 0

            # print("Learner {} beginning!".format(learner_comm.rank))
            lr_stime = MPI.Wtime()
            lr0_sendtime = 0.
            lrs_recvtime = 0.
            tmp = 0.
            if learner_comm.rank == 0:
                # Counter to check number of horovod train steps
                hvd_counter = 0
                send_counter = 0
                recv_counter = 0
                hvd_traintime = 0.

                start = MPI.Wtime()
                worker_episodes = np.arange(1, actor_procs + 1)
                # lr_recv_counter = dict()
                # lr_send_counter = dict()
                # recv_data = None
                # recv_data = np.zeros(learner_procs)
                recv_data = [[]] * learner_procs
                dest_ranks = np.zeros(learner_procs, dtype=int)
                logger.debug('worker_episodes:{}'.format(worker_episodes))

                logger.info("Initializing ...\n")
                # Send initial episode numbers to all actors
                for s in range(learner_procs, agent_comm.size):
                    # Send target weights
                    indices, loss = None, None
                    rank0_epsilon = workflow.agent.epsilon
                    target_weights = workflow.agent.get_weights()
                    episode = worker_episodes[s - learner_procs]
                    agent_comm.send(
                        [episode, rank0_epsilon, target_weights, indices, loss], dest=s)
                    send_counter += 1
                    # lr_send_counter[s] = lr_send_counter.get(s,0)+1

                init_nepisodes = episode
                logger.debug('init_nepisodes:{}'.format(init_nepisodes))

            # Broadcast necessary information to other learners
            indices = learner_comm.bcast(indices, root=0)
            loss = learner_comm.bcast(loss, root=0)
            rank0_epsilon = learner_comm.bcast(rank0_epsilon, root=0)
            target_weights = learner_comm.bcast(target_weights, root=0)

            logger.debug("Learner {} Continuing ...\n".format(learner_comm.rank))
            while episode_done < workflow.nepisodes:
                # print("Running scheduler/learner episode: {}".format(episode))

                if learner_comm.rank == 0:
                    # Receive the rank of the worker ready for more work
                    recv_data[index] = agent_comm.recv(source=MPI.ANY_SOURCE)
                    recv_counter += 1
                    # tmp = agent_comm.recv(source=MPI.ANY_SOURCE)
                    # print(len(tmp))
                    # print(tmp)
                    # recv_data[index] = tmp
                    whofrom = recv_data[index][0]
                    # lr_recv_counter[whofrom] = lr_recv_counter.get(whofrom,0)+1
                    dest_ranks[index] = whofrom
                    step = recv_data[index][1]
                    batch = recv_data[index][2]
                    policy_type = recv_data[index][3]
                    done = recv_data[index][4]
                    logger.debug('step:{}'.format(step))
                    logger.debug('done:{}'.format(done))
                    counter += 1
                    # print("ML Async: counter = {} Actor = {}".format(counter,whofrom))
                    index = counter % learner_procs

                counter = learner_comm.bcast(counter, root=0)
                # print("Learner {} counter value = {}".format(learner_comm.rank,counter))

                # Check if enough data is received for all learners
                if (counter % learner_procs == 0):
                    if learner_comm.rank == 0:
                        # Send the data to all the other learners
                        # print("Printing Receive data!")
                        # print(recv_data)
                        data_buff = recv_data[0]
                        tmp = MPI.Wtime()
                        for s in range(1, learner_comm.size):
                            learner_comm.send(recv_data[s], dest=s)
                        # lr0_sendtime += MPI.Wtime() - tmp
                    else:
                        tmp = MPI.Wtime()
                        data_buff = learner_comm.recv(source=0)
                        step = data_buff[1]
                        batch = data_buff[2]
                        policy_type = data_buff[3]
                        done = data_buff[4]

                    # Synchronize before starting the training
                    learner_comm.Barrier()
                    if learner_comm.rank == 0:
                        lr0_sendtime += MPI.Wtime() - tmp
                    else:
                        lrs_recvtime += MPI.Wtime() - tmp
                    # print("Learner {} data_buff {}".format(learner_comm.rank,data_buff))
                    if learner_comm.rank == 0:
                        # print("Starting training!")
                        s_time = MPI.Wtime()
                    train_return = workflow.agent.train(batch)

                    if learner_comm.rank == 0:
                        # hvd_counter += 1
                        hvd_traintime += MPI.Wtime() - s_time
                        hvd_counter += 1
                        # print("ML Async: Time taken to train (horovod) is {}. No of hvd trains = {}".format(MPI.Wtime()-s_time,hvd_counter))

                    if train_return is not None:
                        if not np.array_equal(train_return[0], (-1 * np.ones(workflow.agent.batch_size))):
                            indices, loss = train_return
                            indices = np.array(indices, dtype=np.intc)
                            loss = np.array(loss, dtype=np.float64)

                    # TODO: Double check if this is already in the DQN code
                    workflow.agent.target_train()
                    if policy_type == 0:
                        workflow.agent.epsilon_adj()
                    epsilon = workflow.agent.epsilon

                    # Send target weights
                    logger.debug('Learner {} rank0_epsilon:{}'.format(learner_comm.rank, epsilon))

                    if learner_comm.rank == 0:
                        target_weights = workflow.agent.get_weights()
                        with open('target_weights.pkl', 'wb') as f:
                            pickle.dump(target_weights, f)

                # Increment episode when starting
                if learner_comm.rank == 0:
                    if step == 0:
                        episode += 1
                        logger.debug('if episode:{}'.format(episode))

                    # Increment the number of completed episodes
                    if done:
                        episode_done += 1
                        latest_episode = worker_episodes.max()
                        worker_episodes[whofrom - learner_procs] = latest_episode + 1
                        logger.debug('episode_done:{}'.format(episode_done))

                    if counter % learner_procs == 0:
                        for dest in dest_ranks:
                            # print("Comm debug: learner to actor: {} --> {}".format(0,dest))
                            # lr_send_counter[dest] = lr_send_counter.get(dest,0)+1
                            agent_comm.send([worker_episodes[dest - learner_procs],
                                             epsilon, target_weights, indices, loss], dest=dest)
                            send_counter += 1

                episode_done = learner_comm.bcast(episode_done, root=0)
                # print("Learner {} episode_done value = {}".format(learner_comm.rank,episode_done))

            # Check if there are any actors waiting
            rem = counter % learner_procs

            if rem != 0:
                # train for one last time
                if learner_comm.rank == 0:
                    # Send the data to all the other learners
                    # print("Printing Receive data!")
                    # print(recv_data)
                    print("Remaining data chunks received: {}".format(rem))
                    data_buff = recv_data[0]
                    tmp = MPI.Wtime()
                    for s in range(1, learner_comm.size):
                        # print("Comm debug: learner to learner {} --> {}".format(0,s))
                        learner_comm.send(recv_data[s], dest=s)
                else:
                    tmp = MPI.Wtime()
                    data_buff = learner_comm.recv(source=0)
                    step = data_buff[1]
                    batch = data_buff[2]
                    policy_type = data_buff[3]
                    done = data_buff[4]

                # Synchronize before starting the training
                learner_comm.Barrier()

                if learner_comm.rank == 0:
                    lr0_sendtime += MPI.Wtime() - tmp

                else:
                    lrs_recvtime += MPI.Wtime() - tmp
                # print("Learner {} data_buff {}".format(learner_comm.rank,data_buff))
                if learner_comm.rank == 0:
                    #    print("Starting training!")
                    s_time = MPI.Wtime()
                train_return = workflow.agent.train(batch)

                if learner_comm.rank == 0:
                    hvd_traintime += MPI.Wtime() - s_time
                    hvd_counter += 1
                #    print("ML Async training time {} for episode {}".format(MPI.Wtime()-s_time,episode_done))

                if train_return is not None:
                    if not np.array_equal(train_return[0], (-1 * np.ones(workflow.agent.batch_size))):
                        indices, loss = train_return
                        indices = np.array(indices, dtype=np.intc)
                        loss = np.array(loss, dtype=np.float64)

                # TODO: Double check if this is already in the DQN code
                workflow.agent.target_train()
                if policy_type == 0:
                    workflow.agent.epsilon_adj()
                epsilon = workflow.agent.epsilon

                # Send target weights
                logger.debug('Learner {} rank0_epsilon:{}'.format(learner_comm.rank, epsilon))

                if learner_comm.rank == 0:
                    target_weights = workflow.agent.get_weights()
                    with open('target_weights.pkl', 'wb') as f:
                        pickle.dump(target_weights, f)

                if learner_comm.rank == 0:
                    for i in range(rem):

                        # print("Comm debug: learner to actor: {} --> {}".format(0,dest_ranks[i]))
                        agent_comm.send([worker_episodes[dest_ranks[i] - learner_procs],
                                         epsilon, target_weights, indices, loss], dest=dest_ranks[i])
                        send_counter += 1
                        # lr_send_counter[dest_ranks[i]] = lr_send_counter.get(dest_ranks[i],0)+1

            # Sai Chenna - Learner 0 will take care of finishing up
            # Will not be using other learners to train as the number of new batches received are not guaranteed to be exact multiple
            # of total number of learners
            if learner_comm.rank == 0:
                # print("Finishing up!")
                logger.info("Finishing up ...\n")
                episode = -1
                for s in range(learner_procs, agent_comm.size):
                    recv_data = agent_comm.recv(source=MPI.ANY_SOURCE)
                    whofrom = recv_data[0]
                    # lr_recv_counter[dest] = lr_recv_counter.get(whofrom,0)+1
                    step = recv_data[1]
                    batch = recv_data[2]
                    epsilon = recv_data[3]
                    done = recv_data[4]
                    logger.debug('step:{}'.format(step))
                    logger.debug('done:{}'.format(done))
                    # Train
                    # train_return = workflow.agent.train(batch)
                    # if train_return is not None:
                    #    indices, loss = train_return
                    # workflow.agent.target_train()
                    # workflow.agent.save(workflow.results_dir + '/model.pkl')
                    # print("Comm debug: learner to actor: {} --> {} ".format(0,s))
                    agent_comm.send([episode, 0, 0, indices, loss], dest=s)
                    # lr_send_counter[s] = lr_send_counter.get(s,0)+1

                logger.info('Learner time: {}'.format(MPI.Wtime() - start))

                # print("Learner 0 Send counter")
                # print(lr_send_counter)
                # print("Learner 0 Receive counter")
                # print(lr_recv_counter)

            print("Learner {} : Total time: {}".format(learner_comm.rank, MPI.Wtime() - lr_stime))
            if learner_comm.rank == 0:
                print("Learner 0 : Total number of data batches received from actors : {}".format(recv_counter + len(range(learner_procs, agent_comm.size))))
                print("Learner 0 : Total number of data batches sent to actors : {}".format(send_counter + len(range(learner_procs, agent_comm.size))))
                print("Learner 0 : Total time spent on training : {}".format(hvd_traintime))
                print("Learner 0 : Total time spent sending data to other learners : {}".format(lr0_sendtime))
                print("Learner 0 : Total horovod trainings done : {}".format(hvd_counter))
                print("Learner 0 : Training throughput : {} batches trained/sec".format((hvd_counter * learner_procs) / hvd_traintime))

            else:
                print("Learner {} : Total time spent receiving batch data from Learner 0 : {}".format(learner_comm.rank, lrs_recvtime))
            # print("Learner {} exited successfully!".format(learner_comm.rank))
            workflow.agent.learner_training_metrics()

        else:
            if mpi_settings.is_actor():
                # Setup logger
                filename_prefix = 'ExaLearner_' + 'Episodes%s_Steps%s_Rank%s_memory_v1' \
                    % (str(workflow.nepisodes), str(workflow.nsteps), str(agent_comm.rank))
                train_file = open(workflow.results_dir + '/' +
                                  filename_prefix + ".log", 'w')
                train_writer = csv.writer(train_file, delimiter=" ")
                ac_recvtime = 0.
                ac_sendtime = 0.
                tmp = 0.
                ac_send_counter = 0
                ac_recv_counter = 0

            start = MPI.Wtime()
            while episode != -1:
                # Reset variables each episode
                workflow.env.seed(0)
                # TODO: optimize some of these variables out for env processes
                current_state = workflow.env.reset()
                total_reward = 0
                steps = 0
                action = 0

                # images = []
                # filename = "cartpole-demo-episode{}.gif".format(episode)

                # Steps in an episode
                while steps < workflow.nsteps:
                    # logger.debug('ASYNC::run() agent_comm.rank{}; step({} of {})'
                    #             .format(agent_comm.rank, steps, (workflow.nsteps - 1)))
                    if mpi_settings.is_actor():
                        # Receive target weights
                        # print("Actor with Rank {} total number of msgs received from Rank 0 = {}".format(agent_comm.rank,actr_recv_counter))
                        # print("Actor with Rank {} total number of msgs sent to Rank 0 = {}".format(agent_comm.rank,actr_send_counter))
                        tmp = MPI.Wtime()
                        recv_data = agent_comm.recv(source=0)
                        ac_recvtime += MPI.Wtime() - tmp
                        ac_recv_counter += 1
                        # actr_recv_counter += 1

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
                        workflow.agent.epsilon = recv_data[1]
                        workflow.agent.set_weights(recv_data[2])

                        if workflow.action_type == 'fixed':
                            action, policy_type = 0, -11
                        else:
                            action, policy_type = workflow.agent.action(
                                current_state)

                    next_state, reward, done, _ = workflow.env.step(action)

                    # workflow.env.render(mode='human')

                    # screen = workflow.env.render(mode='rgb_array')
                    # images.append(Image.fromarray(screen))

                    if mpi_settings.is_actor():
                        total_reward += reward
                        memory = (current_state, action, reward,
                                  next_state, done, total_reward)

                        with open('experience.pkl', 'wb') as f:
                            pickle.dump(memory, f)
                        # batch_data = []
                        workflow.agent.remember(
                            memory[0], memory[1], memory[2], memory[3], memory[4])

                        # s_gendata = MPI.Wtime()
                        batch_data = next(workflow.agent.generate_data())
                        # print("Time taken to generate data(serially) of batch size %s on 1 actor rank is %s)"
                        # % (str(workflow.agent.batch_size),str(MPI.Wtime()-s_gendata)))
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

                    if mpi_settings.is_actor():
                        tmp = MPI.Wtime()
                        agent_comm.send(
                            [agent_comm.rank, steps, batch_data, policy_type, done], dest=0)
                        ac_sendtime += MPI.Wtime() - tmp
                        ac_send_counter += 1
                        # actr_send_counter += 1
                        # indices, loss = agent_comm.recv(source=MPI.ANY_SOURCE)
                        indices, loss = recv_data[3:5]
                        if indices is not None:
                            workflow.agent.set_priorities(indices, loss)
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
                    done = env_comm.bcast(done, root=0)
                    # Break for loop if done
                    if done:
                        break

                # images[0].save(filename,save_all=True, append_images=images[1:],loop=0, duration=1)
            logger.info('Worker time = {}'.format(MPI.Wtime() - start))
            if mpi_settings.is_actor():
                train_file.close()

        if mpi_settings.is_actor():
            logger.info(f'Agent[{agent_comm.rank}] timing info:\n')
            workflow.agent.print_timers()
            print("Actor {} :  Total time : {}".format(agent_comm.rank, MPI.Wtime() - start))
            print("Actor {} : Time spent sending data to learner 0 : {}".format(agent_comm.rank, ac_sendtime))
            print("Actor {} : Time spent receiving data to learner 0 : {}".format(agent_comm.rank, ac_recvtime))
            print("Actor {} : Total batches sent to learner 0 : {}".format(agent_comm.rank, ac_send_counter))
            print("Actor {} : Total model data received from learner 0 : {}".format(agent_comm.rank, ac_recv_counter))

            # print("Actor with Rank = {} exited successfully!".format(agent_comm.rank))
