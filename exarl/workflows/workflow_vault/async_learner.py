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
import sys
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, GaussianNoise, BatchNormalization, Flatten, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2, l1_l2
import tensorflow as tf

class ASYNC(erl.ExaWorkflow):
    def __init__(self):
        print('Creating ASYNC learner workflow...')
        #Sai Chenna - adding the following variables to faciltate accelerating generate_data in DQN agent
        self.gamma = []
        self.target_model = []
        self.device = []

        self.agent = cd.run_params["agent"]
        self.accelerate_datagen = False
        #Sai Chenna: do this only for DQN agent
        if self.agent == 'DQN-v0':
            self.accelerate_datagen = True
            #self.accelerate_datagen = False

        if self.agent == 'DQN-v0' and self.accelerate_datagen:
            self.model_type = cd.run_params["model_type"]
            if self.model_type == 'MLP':
                # for mlp
                self.dense = cd.run_params["dense"]

            if self.model_type == 'LSTM':
                # for lstm
                self.lstm_layers = cd.run_params["lstm_layers"]
                self.gauss_noise = cd.run_params["gauss_noise"]
                self.regularizer = cd.run_params["regularizer"]
                self.clipnorm = cd.run_params["clipnorm"]
                self.clipvalue = cd.run_params["clipvalue"]

            # for both
            self.activation = cd.run_params["activation"]
            self.out_activation = cd.run_params["out_activation"]
            self.optimizer = cd.run_params["optimizer"]
            self.loss = cd.run_params["loss"]



    @PROFILE
    def run(self, workflow):

        # MPI communicators
        agent_comm = mpi_settings.agent_comm
        env_comm = mpi_settings.env_comm

        if self.agent == 'DQN-v0' and self.accelerate_datagen:
            #Sai Chenna - build target model on all process except learner
            if not mpi_settings.is_learner():
                if self.model_type == 'MLP':
                    layers = []
                    state_input = Input(shape=(1, workflow.env.observation_space.shape[0]))
                    layers.append(state_input)
                    #self.dense = cd.run_params["dense"]
                    length = len(self.dense)
                    # for i, layer_width in enumerate(self.dense):
                    for i in range(length):
                        layer_width = self.dense[i]
                        layers.append(Dense(layer_width, activation=self.activation)(layers[-1]))
                    # output layer
                    layers.append(Dense(workflow.env.action_space.n, activation=self.out_activation)(layers[-1]))
                    layers.append(Flatten()(layers[-1]))

                    self.target_model = Model(inputs=layers[0], outputs=layers[-1])

                elif self.model_type == 'LSTM':

                    num_layers = len(self.lstm_layers)

                    self.target_model = Sequential()
                    # special case for input layer
                    self.target_model.add(LSTM(self.lstm_layers[0], activation=self.activation,return_sequences=True, input_shape=(1, workflow.env.observation_space.shape[0])))
                    self.target_model.add(BatchNormalization())
                    self.target_model.add(Dropout(self.gauss_noise[0]))

                    # loop over inner layers only
                    for l in range(1, num_layers - 1):
                        self.target_model.add(LSTM(self.lstm_layers[l], activation=self.activation,return_sequences=True))
                        self.target_model.add(Dropout(self.gauss_noise[l]))

                    # special case for output layer
                    l = num_layers = 1
                    self.target_model.add(LSTM(self.lstm_layers[l], activation=self.activation,kernel_regularizer=l1_l2(self.regularizer[0], self.regularizer[1]),))
                    self.target_model.add(Dropout(self.gauss_noise[l]))
                    self.target_model.add(Dense(workflow.env.action_space.n, activation=self.out_activation))

                else:
                    sys.exit("Oops! That was not a valid model type. Try again...")


        #if(env_comm.rank != 0):
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

        inference_time = 0.0
        inference_nb = 0
        fixed_action =  [np.array(0.0)]
        # Round-Robin Scheduler
        if mpi_settings.is_learner():
            start = MPI.Wtime()
            recv_counter = 0
            send_counter = 0
            train_counter = 0
            training_time = 0.
            # worker_episodes = np.linspace(0, agent_comm.size - 2, agent_comm.size - 1)
            worker_episodes = np.arange(1, agent_comm.size)
            logger.debug('worker_episodes:{}'.format(worker_episodes))

            logger.info("Initializing ...\n")
            for s in range(1, agent_comm.size):
                # Send target weights
                indices, loss = None, None
                rank0_epsilon = workflow.agent.epsilon
                target_weights = workflow.agent.get_weights()
                episode = worker_episodes[s - 1]
                agent_comm.send(
                    [episode, rank0_epsilon, target_weights, indices, loss], dest=s)
                send_counter += 1

            init_nepisodes = episode
            logger.debug('init_nepisodes:{}'.format(init_nepisodes))

            logger.debug("Continuing ...\n")
            while episode_done < workflow.nepisodes:
                # print("Running scheduler/learner episode: {}".format(episode))

                # Receive the rank of the worker ready for more work
                recv_data = agent_comm.recv(source=MPI.ANY_SOURCE)
                recv_counter += 1

                whofrom = recv_data[0]
                step = recv_data[1]
                batch = recv_data[2]
                policy_type = recv_data[3]
                done = recv_data[4]
                logger.debug('step:{}'.format(step))
                logger.debug('done:{}'.format(done))
                # Train
                train_stime = MPI.Wtime()
                train_return = workflow.agent.train(batch)
                training_time += MPI.Wtime() - train_stime
                train_counter += 1
                if train_return is not None:
                    if not np.array_equal(train_return[0], (-1 * np.ones(workflow.agent.batch_size))):
                        indices, loss = train_return

                # agent_comm.send([indicies, loss], dest=whofrom)

                # TODO: Double check if this is already in the DQN code
                workflow.agent.target_train()
                if policy_type == 0:
                    workflow.agent.epsilon_adj()
                epsilon = workflow.agent.epsilon

                # Send target weights
                logger.debug('rank0_epsilon:{}'.format(epsilon))

                target_weights = workflow.agent.get_weights()
                with open('target_weights.pkl', 'wb') as f:
                    pickle.dump(target_weights, f)

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
                                 epsilon, target_weights, indices, loss], dest=whofrom)
                send_counter += 1

            logger.info("Finishing up ...\n")
            episode = -1
            for s in range(1, agent_comm.size):
                recv_data = agent_comm.recv(source=MPI.ANY_SOURCE)
                recv_counter += 1
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
                workflow.agent.save(workflow.results_dir + '/model.pkl')
                agent_comm.send([episode, 0, 0, indices, loss], dest=s)
                send_counter += 1

            logger.info('Learner time: {}'.format(MPI.Wtime() - start))
            print('Learner {} time: {}'.format(0,MPI.Wtime() - start))
            print("Learner {} Total times data received from actors: {}".format(0,recv_counter))
            print("Learner {} Total times data sent to actors: {}".format(0,send_counter))
            print("Learner {} No of trainings done: {}".format(0,train_counter))
            print("Learner {} Total time spent on training: {}".format(0,training_time))
            print("Learner {} Training throughput: {} trainings/sec".format(0,train_counter/training_time))
            workflow.agent.learner_training_metrics()

        else:
            if mpi_settings.is_actor():
                # Setup logger
                filename_prefix = 'ExaLearner_' + 'Episodes%s_Steps%s_Rank%s_memory_v1' \
                    % (str(workflow.nepisodes), str(workflow.nsteps), str(agent_comm.rank))
                train_file = open(workflow.results_dir + '/' +
                                  filename_prefix + ".log", 'w')
                train_writer = csv.writer(train_file, delimiter=" ")
                ac_sendtime = 0.
                ac_recvtime = 0.
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

                # Steps in an episode
                while steps < workflow.nsteps:
                    #logger.debug('ASYNC::run() agent_comm.rank{}; step({} of {})'
                                #.format(agent_comm.rank, steps, (workflow.nsteps - 1)))
                    if mpi_settings.is_actor():
                        # Receive target weights
                        tmp = MPI.Wtime()
                        recv_data = agent_comm.recv(source=0)
                        ac_recvtime += MPI.Wtime() - tmp
                        ac_recv_counter += 1
                        # Update episode while beginning a new one i.e. step = 0
                        if steps == 0:
                            episode = recv_data[0]
                            # print(episode)
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
                        inference_time -= MPI.Wtime()
                        if workflow.action_type == 'fixed':
                            #action, policy_type = 0, -11
                            action, policy_type = fixed_action, 1
                        else:
                            action, policy_type = workflow.agent.action(current_state)
                        inference_time += MPI.Wtime()
                        inference_nb += 1
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

                        #Sai Chenna -normal operation if either agent is not DQN or if accelerating data pipeline is not enabled in DQN agent
                        if self.agent != 'DQN-v0' or self.accelerate_datagen == False :
                            s_gendata = MPI.Wtime()
                            batch_data = next(workflow.agent.generate_data())
                            #e_gendata = MPI.Wtime()
                            #print("Time taken to generate data(serially) of batch size %s on 1 actor rank is %s)" % (str(workflow.agent.batch_size),str(MPI.Wtime()-s_gendata)))
                            logger.info(
                                'Rank[{}] - Generated data: {}'.format(agent_comm.rank, len(batch_data[0])))
                            try:
                                buffer_length = len(workflow.agent.memory)
                            except:
                                buffer_length = workflow.agent.replay_buffer.get_buffer_length()
                            logger.info(
                                'Rank[{}] - # Memories: {}'.format(agent_comm.rank, buffer_length))


                    #Sai Chenna - accelerate generate data pipeline of DQN agent if the flag is set to TRUE
                    if self.agent == 'DQN-v0' and self.accelerate_datagen:
                        batch_data = self.get_data_parallel(workflow)


                    if steps >= workflow.nsteps - 1:
                        done = True

                    if mpi_settings.is_actor():
                        # Send batched memories
                        tmp = MPI.Wtime()
                        agent_comm.send(
                            [agent_comm.rank, steps, batch_data, policy_type, done], dest=0)
                        # indices, loss = agent_comm.recv(source=MPI.ANY_SOURCE)
                        ac_sendtime += MPI.Wtime() - tmp
                        ac_send_counter += 1
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
            logger.info('Worker time = {}'.format(MPI.Wtime() - start))
            if mpi_settings.is_actor():
                train_file.close()

                print("[{}] total_inference_time : {} , total_inference_nb : {}, inferences/sec {}".format(agent_comm.rank,inference_time, inference_nb, inference_nb/inference_time))

        if mpi_settings.is_actor():
            logger.info(f'Agent[{agent_comm.rank}] timing info:\n')
            workflow.agent.print_timers()
            print("Actor {} :  Total time : {}".format(agent_comm.rank,MPI.Wtime() - start))
            print("Actor {} : Time spent sending data to learner 0 : {}".format(agent_comm.rank,ac_sendtime))
            print("Actor {} : Time spent receving data to learner 0 : {}".format(agent_comm.rank,ac_recvtime))
            print("Actor {} : Total batches sent to learner 0 : {}".format(agent_comm.rank,ac_send_counter))
            print("Actor {} : Total model data received from learner 0 : {}".format(agent_comm.rank,ac_recv_counter))


    #Sai Chenna - parallelize generate_data method in DQN agent by distributing batchsize to all processes_per_env
    def get_data_parallel(self,workflow):
        env_comm = mpi_settings.env_comm
        global_rank = mpi_settings.global_comm.rank
        batch_data = []
        first_offset = []
        chunk_size = []
        batch_data_part = []
        early_stop = False
        model_weights = []
        minibatch = []
        my_minibatch = []
        if (env_comm.rank == 0):
            s_gendata_par = MPI.Wtime()
            self.gamma = workflow.agent.gamma
            batch_size = workflow.agent.batch_size
            self.device = workflow.agent.device
            self.target_model = workflow.agent.target_model
            memory_len = len(workflow.agent.memory)
            model_weights = self.target_model.get_weights()
            if (memory_len < batch_size):
                early_stop = True
                batch_states = np.zeros(
                    (batch_size, 1, workflow.env.observation_space.shape[0])
                ).astype("float64")
                batch_target = np.zeros((batch_size, workflow.env.action_space.n)).astype(
                    "float64"
                )
                batch_data = batch_states,batch_target
                #print("Early stop Time taken to generate data(parallely) of batch size %s on %s ranks is %s)" % (str(batch_size),str(env_comm.size),str(MPI.Wtime()-s_gendata_par)))
            else:
                first_offset = int(batch_size/env_comm.size) + (batch_size%env_comm.size)
                chunk_size =  int(batch_size/env_comm.size)
                minibatch = workflow.agent.get_minibatch()

        early_stop = env_comm.bcast(early_stop,root=0)
        model_weights = env_comm.bcast(model_weights,root=0)
        if (early_stop == True):
            batch_data = env_comm.bcast(batch_data,root=0)
            #if (env_comm.rank == 0):
                #print("Early stop Time taken to generate data(parallely) of batch size %s on %s ranks is %s)" % (str(batch_size),str(env_comm.size),str(MPI.Wtime()-s_gendata_par)))
            return batch_data

        self.gamma = env_comm.bcast(self.gamma,root=0)
        self.device = env_comm.bcast(self.device,root=0)
        first_offset = env_comm.bcast(first_offset,root=0)
        chunk_size = env_comm.bcast(chunk_size,root=0)
        minibatch = env_comm.bcast(minibatch,root=0)
        if (env_comm.rank != 0):
            self.target_model.set_weights(model_weights)

        if(env_comm.rank == 0):
            my_minibatch = minibatch[:first_offset]
        else:
            my_minibatch = minibatch[int(first_offset+((env_comm.rank-1)*chunk_size)):int(first_offset+((env_comm.rank)*chunk_size))]

        #generaate the training data on each processes
        batch_target = list(map(self.calc_target_f_parallel, my_minibatch))
        try:

            batch_states = [np.array(exp[0]).reshape(1, 1, len(exp[0]))[0] for exp in my_minibatch]
            batch_states = np.reshape(batch_states, [len(my_minibatch), 1, len(my_minibatch[0][0])]).astype("float64")
            batch_target = np.reshape(batch_target, [len(my_minibatch), workflow.env.action_space.n]).astype("float64")
        except:
            print("Global Rank: %s Environment Local Rank = %s Length of my mini-batch: %s " % (str(global_rank),str(env_comm.rank),str(len(my_minibatch))))
            print("My mini-batch: %s" % (str(my_minibatch)))
            print("Size of the original minibatch: %s" % (str(len(minibatch))))
            print("Original minibatch: %s" % (str(minibatch)))


        batch_states = env_comm.gather(batch_states,root=0)
        batch_target = env_comm.gather(batch_target,root=0)


        #if (env_comm.rank == 0):
            #print("Time taken to generate data(parallely) of batch size %s on %s ranks is %s)" % (str(batch_size),str(env_comm.size),str(MPI.Wtime()-s_gendata_par)))


        return batch_states, batch_target

    def calc_target_f_parallel(self,exp):
        state, action, reward, next_state, done = exp
        np_state = np.array(state).reshape(1, 1, len(state))
        np_next_state = np.array(next_state).reshape(1, 1, len(next_state))
        expectedQ = 0
        if not done:
            with tf.device(self.device):
                expectedQ = self.gamma * np.amax(
                    self.target_model.predict(np_next_state)[0]
                )
        target = reward + expectedQ
        with tf.device(self.device):
            target_f = self.target_model.predict(np_state)
        target_f[0][action] = target
        return target_f[0]
