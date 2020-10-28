import mpi4py.rc; mpi4py.rc.threads = False
from mpi4py import MPI
import random,sys,os, time
import numpy as np
from datetime import datetime
from collections import deque
from tensorflow.python.client import device_lib
from keras import backend as K
import csv,json,math
import exarl as erl
import pickle
import exarl.mpi_settings as mpi_settings
import sys
import tensorflow as tf
tf_version = int((tf.__version__)[0])
from tensorflow.compat.v1.keras.backend import set_session

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RL-Logger')
logger.setLevel(logging.INFO)

# The Deep Q-Network (DQN)
class DQN(erl.ExaAgent):
    def __init__(self, env):
        #
        self.is_learner = False
        self.model = None
        self.target_model = None
        self.target_weights = None

        self.env = env
        self.agent_comm = mpi_settings.agent_comm

        # MPI
        self.rank = self.agent_comm.rank
        self.size = self.agent_comm.size

        self._get_device()
        #self.device = '/CPU:0'
        logger.info('Using device: {}'.format(self.device))
        #tf.config.experimental.set_memory_growth(self.device, True)

        # Default settings
        num_cores = os.cpu_count()
        num_CPU = os.cpu_count()
        num_GPU = 0

        # Setup GPU cfg
        # if tf_version < 2:
        #     gpu_names = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
        #     if self.rank==0 and len(gpu_names)>0:
        #             num_cores = 1
        #             num_CPU = 1
        #             num_GPU = len(gpu_names)
        #     config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
        #                 inter_op_parallelism_threads=num_cores,
        #                 allow_soft_placement=True,
        #                 device_count = {'CPU' : num_CPU,
        #                                 'GPU' : num_GPU})
        #     config.gpu_options.allow_growth = True
        #     sess = tf.Session(config=config)
        #     set_session(sess)
        # elif tf_version >= 2:
        #
        #     config = tf.compat.v1.ConfigProto()
        #     config.gpu_options.allow_growth = True
        #     sess = tf.compat.v1.Session(config=config)
        #     tf.compat.v1.keras.backend.set_session(sess)

        # Declare hyper-parameters, initialized for determining datatype
        super().__init__()
        self.results_dir = ''
        self.search_method = ''
        self.gamma = 0.0
        self.epsilon = 0.0
        self.epsilon_min = 0.0
        self.epsilon_decay = 0.0
        self.learning_rate = 0.0
        self.batch_size = 0
        self.tau = 0.0
        self.model_type = ''

        # for mlp
        self.dense = [0, 0]

        # for lstm
        self.lstm_layers = [0, 0]
        self.gauss_noise = [0.0, 0.0]
        self.regularizer = [0.0, 0.0]

        # for both
        self.activation = 'relu'
        self.out_activation = 'relu'
        self.optimizer = 'adam'
        self.loss = 'mse'
        self.clipnorm = 1.0
        self.clipvalue = 0.5

        # TODO: make configurable
        self.memory = deque(maxlen=1000)

    def _get_device(self):
        #cpus = tf.config.experimental.list_physical_devices('CPU')
        gpus = tf.config.experimental.list_physical_devices('GPU')
        ngpus = len(gpus)
        logging.info('Number of available GPUs: {}'.format(ngpus))
        if ngpus > 0:
            gpu_id = self.rank % ngpus
            self.device = '/GPU:{}'.format(gpu_id)
            tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
        else:
            self.device = '/CPU:0'

    def set_agent(self):
        # Get hyper-parameters
        agent_data = super().get_config()

        # dqn intrinsic variables
        self.results_dir = agent_data['output_dir']
        self.gamma = agent_data['gamma']
        self.epsilon = agent_data['epsilon']
        self.epsilon_min = agent_data['epsilon_min']
        self.epsilon_decay = agent_data['epsilon_decay']
        self.learning_rate = agent_data['learning_rate']
        self.batch_size = agent_data['batch_size']
        self.tau = agent_data['tau']
        self.model_type = agent_data['model_type']

        # for mlp
        self.dense = agent_data['dense']

        # for lstm
        self.lstm_layers = agent_data['lstm_layers']
        self.gauss_noise = agent_data['gauss_noise']
        self.regularizer = agent_data['regularizer']

        # for both
        self.activation = agent_data['activation']
        self.out_activation = agent_data['out_activation']
        self.optimizer = agent_data['optimizer']
        self.clipnorm = agent_data['clipnorm']
        self.clipvalue = agent_data['clipvalue']

        # Build network model
        with tf.device(self.device):
            if self.is_learner:
                self.model = self._build_model()

        with tf.device('/CPU:0'):
            self.target_model = self._build_model()
            self.target_weights = self.target_model.get_weights()

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

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def action(self, state):
        random.seed(datetime.now())
        random_data = os.urandom(4)
        np.random.seed(int.from_bytes(random_data, byteorder="big"))
        rdm = np.random.rand()
        if rdm <= self.epsilon:
            action = random.randrange(self.env.action_space.n)
            return action, 0
        else:
            np_state = np.array(state).reshape(1,1,len(state))
            with tf.device(self.device):
                act_values = self.target_model.predict(np_state)
            action = np.argmax(act_values[0])
            return action, 1

    def play(self, state):
        act_values = self.target_model.predict(state)
        return np.argmax(act_values[0])

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

    def generate_data(self):
        # Worker method to create samples for training
        # TODO: This method is the most expensive and takes 90% of the agent compute time
        # TODO: Reduce computational time
        batch_states = []
        batch_target = []
        # Return empty batch
        if len(self.memory)<self.batch_size:
            yield batch_states, batch_target
        start_time_episode = time.time()
        minibatch = random.sample(self.memory, self.batch_size)
        logger.debug('Agent - Minibatch time: %s ' % (str(time.time() - start_time_episode)))
        batch_target = list(map(self.calc_target_f, minibatch))
        batch_states = [np.array(exp[0]).reshape(1,1,len(exp[0]))[0] for exp in minibatch]
        batch_states = np.reshape(batch_states, [len(minibatch), 1, len(minibatch[0][0])])
        batch_target = np.reshape(batch_target, [len(minibatch), self.env.action_space.n])
        yield batch_states, batch_target

    def train(self, batch):
        self.epsilon_adj()
        if self.is_learner:
            # if len(self.memory) > (self.batch_size) and len(batch_states)>=(self.batch_size):
            if len(batch[0])>=(self.batch_size):
                # batch_states, batch_target = batch
                start_time_episode = time.time()
                with tf.device(self.device):
                    history = self.model.fit(batch[0], batch[1], epochs=1, verbose=0)
                logger.info('Agent[%s]- Training: %s ' % (str(self.rank), str(time.time() - start_time_episode)))
                start_time_episode = time.time()
                logger.info('Agent[%s] - Target update time: %s ' % (str(self.rank), str(time.time() - start_time_episode)))
        else:
            logger.warning('Training will not be done because this instance is not set to learn.')

    def get_weights(self):
        logger.debug('Agent[%s] - get target weight.' % str(self.rank))
        return self.target_model.get_weights()

    def set_weights(self, weights):
        logger.info('Agent[%s] - set target weight.' % str(self.rank))
        logger.debug('Agent[%s] - set target weight: %s' % (str(self.rank),weights))
        with tf.device(self.device):
            self.target_model.set_weights(weights)

    def target_train(self):
        if self.is_learner:
            logger.info('Agent[%s] - update target weights.' % str(self.rank))
            model_weights = self.model.get_weights()
            target_weights = self.target_model.get_weights()
            for i in range(len(target_weights)):
                target_weights[i] = self.tau*model_weights[i] + (1-self.tau)*target_weights[i]
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

    def update(self):
        logger.info("Implement update method in dqn.py")

    def monitor(self):
        logger.info("Implement monitor method in dqn.py")

    def benchmark(dataset, num_epochs=1):
        start_time = time.perf_counter()
        for epoch_num in range(num_epochs):
            for sample in dataset:
                # Performing a training step
                time.sleep(0.01)
                print(sample)
        tf.print("Execution time:", time.perf_counter() - start_time)
