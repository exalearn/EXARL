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
import os
import math
import json
import csv
import random
import tensorflow as tf
import sys
import pickle
import exarl as erl
from exarl.comm_base import ExaComm
from keras import backend as K
from tensorflow.python.client import device_lib
from collections import deque
from datetime import datetime
import numpy as np
import utils.candleDriver as cd
import utils.log as log
from utils.introspect import introspectTrace
from tensorflow.compat.v1.keras.backend import set_session

tf_version = int((tf.__version__)[0])

logger = log.setup_logger(__name__, cd.run_params['log_level'])

# The Deep Q-Network (DQN)


class DQN(erl.ExaAgent):
    def __init__(self, env):
        #
        self.is_learner = False
        self.model = None
        self.target_model = None
        self.target_weights = None

        self.env = env
        self.agent_comm = ExaComm.agent_comm

        # MPI
        self.rank = self.agent_comm.rank
        self.size = self.agent_comm.size

        self._get_device()
        logger.info('Using device: {}'.format(self.device))

        # Timers
        self.training_time = 0
        self.ntraining_time = 0
        self.dataprep_time = 0
        self.ndataprep_time = 0

        # Default settings
        num_cores = os.cpu_count()
        num_CPU = os.cpu_count()
        num_GPU = 0

        # Setup GPU cfg
        if tf_version < 2:
            gpu_names = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
            if self.rank == 0 and len(gpu_names) > 0:
                num_cores = 1
                num_CPU = 1
                num_GPU = len(gpu_names)
            config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                                    inter_op_parallelism_threads=num_cores,
                                    allow_soft_placement=True,
                                    device_count={'CPU': num_CPU,
                                                  'GPU': num_GPU})
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            set_session(sess)
        elif tf_version >= 2:
            if self.rank == 0:
                print("Setting GPU rank", self.rank)
                config = tf.compat.v1.ConfigProto(device_count={'GPU':1, 'CPU':1})
            else:
                print("Setting no GPU rank", self.rank)
                config = tf.compat.v1.ConfigProto(device_count={'GPU':0, 'CPU':1})
            config.gpu_options.allow_growth = True
            sess = tf.compat.v1.Session(config=config)
            tf.compat.v1.keras.backend.set_session(sess)

        # dqn intrinsic variables
        self.results_dir = cd.run_params['output_dir']
        self.gamma = cd.run_params['gamma']
        self.epsilon = cd.run_params['epsilon']
        self.epsilon_min = cd.run_params['epsilon_min']
        self.epsilon_decay = cd.run_params['epsilon_decay']
        self.learning_rate = cd.run_params['learning_rate']
        self.batch_size = cd.run_params['batch_size']
        self.tau = cd.run_params['tau']
        self.model_type = cd.run_params['model_type']

        # for mlp
        if self.model_type == 'MLP':
            self.dense = cd.run_params['dense']
            self.out_activation = cd.run_params['out_activation']

        # for lstm
        elif self.model_type == 'LSTM':
            self.lstm_layers = cd.run_params['lstm_layers']
            self.gauss_noise = cd.run_params['gauss_noise']
            self.regularizer = cd.run_params['regularizer']
            self.out_activation = cd.run_params['out_activation']

        # for both
        self.activation = cd.run_params['activation']
        self.optimizer = cd.run_params['optimizer']
        self.loss = cd.run_params['loss']

        # Build network model
        if self.is_learner:
            with tf.device(self.device):
                self.model = self._build_model()
                self.model.compile(loss=self.loss, optimizer=self.optimizer)
                self.model.summary()
        else:
            with tf.device('/CPU:0'):
                self.target_model = self._build_model()
                self.target_model.compile(loss=self.loss, optimizer=self.optimizer)
                self.target_model.summary()
                self.target_weights = self.target_model.get_weights()

        # TODO: make configurable
        self.memory = deque(maxlen=1000)

    def _get_device(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        ngpus = len(gpus)
        logger.info('Number of available GPUs: {}'.format(ngpus))
        if ngpus > 0:
            gpu_id = self.rank % ngpus
            self.device = '/GPU:{}'.format(gpu_id)
        else:
            self.device = '/CPU:0'

    def _build_model(self):
        if self.model_type == 'MLP':
            from agents.agent_vault._build_mlp import build_model
            return build_model(self)
        elif self.model_type == 'LSTM':
            from agents.agent_vault._build_lstm import build_model
            return build_model(self)
        else:
            sys.exit("Oops! That was not a valid model type. Try again...")

    def set_learner(self):
        logger.debug('Agent[{}] - Creating active model for the learner'.format(self.rank))
        self.is_learner = True
        self.model = self._build_model()
        self.model.compile(loss=self.loss, optimizer=self.optimizer)
        self.model.summary()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    @introspectTrace()
    def action(self, state):
        random.seed(datetime.now())
        random_data = os.urandom(4)
        np.random.seed(int.from_bytes(random_data, byteorder="big"))
        rdm = np.random.rand()
        if rdm <= self.epsilon:
            self.epsilon_adj()
            action = random.randrange(self.env.action_space.n)
            return action, 0
        else:
            np_state = np.array(state).reshape(1, 1, len(state))
            with tf.device(self.device):
                act_values = self.target_model.predict(np_state)
            action = np.argmax(act_values[0])
            return action, 1

    def play(self, state):
        with tf.device(self.device):
            act_values = self.target_model.predict(state)
        return np.argmax(act_values[0])

    @introspectTrace()
    def calc_target_f(self, exp):
        state, action, reward, next_state, done = exp
        np_state = np.array(state).reshape(1, 1, len(state))
        np_next_state = np.array(next_state).reshape(1, 1, len(next_state))
        expectedQ = 0
        if not done:
            with tf.device(self.device):
                expectedQ = self.gamma * np.amax(self.target_model.predict(np_next_state)[0])
        target = reward + expectedQ
        with tf.device(self.device):
            target_f = self.target_model.predict(np_state)
        target_f[0][action] = target
        return target_f[0]

    @introspectTrace()
    def generate_data(self):
        # Worker method to create samples for training
        # TODO: This method is the most expensive and takes 90% of the agent compute time
        # TODO: Reduce computational time
        # TODO: Revisit the shape (e.g. extra 1 for the LSTM)
        batch_states = np.zeros((self.batch_size, 1, self.env.observation_space.shape[0]))
        batch_target = np.zeros((self.batch_size, self.env.action_space.n))
        # batch_states = []
        # batch_target = []
        # Return empty batch
        if len(self.memory) < self.batch_size:
            yield batch_states, batch_target
        start_time = time.time()
        minibatch = random.sample(self.memory, self.batch_size)
        batch_target = list(map(self.calc_target_f, minibatch))
        batch_states = [np.array(exp[0]).reshape(1, 1, len(exp[0]))[0] for exp in minibatch]
        batch_states = np.reshape(batch_states, [len(minibatch), 1, len(minibatch[0][0])]).astype("float64")
        batch_target = np.reshape(batch_target, [len(minibatch), self.env.action_space.n]).astype("float64")
        end_time = time.time()
        self.dataprep_time += (end_time - start_time)
        self.ndataprep_time += 1
        logger.debug('Agent[{}] - Minibatch time: {} '.format(self.rank, (end_time - start_time)))
        yield batch_states, batch_target

    @introspectTrace()
    def train(self, batch):
        if self.is_learner:
            if len(batch) > 0 and len(batch[0]) >= (self.batch_size):
                start_time = time.time()
                with tf.device(self.device):
                    history = self.model.fit(batch[0], batch[1], epochs=1, verbose=0)
                end_time = time.time()
                self.training_time += (end_time - start_time)
                self.ntraining_time += 1
                logger.info('Agent[{}]- Training: {} '.format(self.rank, (end_time - start_time)))
                start_time_episode = time.time()
                logger.info('Agent[%s] - Target update time: %s ' % (str(self.rank), str(time.time() - start_time_episode)))

                return history
        else:
            logger.warning('Training will not be done because this instance is not set to learn.')

    def get_weights(self):
        logger.debug('Agent[%s] - get target weight.' % str(self.rank))
        return self.target_model.get_weights()

    def set_weights(self, weights):
        logger.info('Agent[%s] - set target weight.' % str(self.rank))
        logger.debug('Agent[%s] - set target weight: %s' % (str(self.rank), weights))
        with tf.device(self.device):
            self.target_model.set_weights(weights)

    def target_train(self):
        if self.is_learner:
            logger.info('Agent[%s] - update target weights.' % str(self.rank))
            with tf.device(self.device):
                model_weights = self.model.get_weights()
                target_weights = self.target_model.get_weights()
            for i in range(len(target_weights)):
                target_weights[i] = self.tau * model_weights[i] + (1 - self.tau) * target_weights[i]
            self.set_weights(target_weights)
        else:
            logger.warning('Weights will not be updated because this instance is not set to learn.')

    def epsilon_adj(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, filename):
        layers = self.target_model.layers
        with open(filename, 'rb') as f:
            pickle_list = pickle.load(f)

        for layerId in range(len(layers)):
            assert(layers[layerId].name == pickle_list[layerId][0])
            layers[layerId].set_weights(pickle_list[layerId][1])

    def save(self, filename):
        layers = self.target_model.layers
        pickle_list = []
        for layerId in range(len(layers)):
            weigths = layers[layerId].get_weights()
            pickle_list.append([layers[layerId].name, weigths])

        with open(filename, 'wb') as f:
            pickle.dump(pickle_list, f, -1)

    def benchmark(dataset, num_epochs=1):
        start_time = time.perf_counter()
        for epoch_num in range(num_epochs):
            for sample in dataset:
                # Performing a training step
                time.sleep(0.01)
                print(sample)
        tf.print("Execution time:", time.perf_counter() - start_time)

    def print_timers(self):
        if self.ntraining_time > 0:
            logger.info("Agent[{}] - Average training time: {}".format(self.rank,
                                                                       self.training_time / self.ntraining_time))
        else:
            logger.info("Agent[{}] - Average training time: {}".format(self.rank, 0))

        if self.ndataprep_time > 0:
            logger.info("Agent[{}] - Average data prep time: {}".format(self.rank,
                                                                        self.dataprep_time / self.ndataprep_time))
        else:
            logger.info("Agent[{}] - Average data prep time: {}".format(self.rank, 0))
