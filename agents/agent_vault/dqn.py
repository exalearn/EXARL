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
import keras as keras
#from keras.backend.tensorflow_backend import set_session
from tensorflow.compat.v1.keras.backend import set_session

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RL-Logger')
logger.setLevel(logging.ERROR)

##
#import multiprocessing

#The Deep Q-Network (DQN)
class DQN(erl.ExaAgent):
    def __init__(self, env):

        #import pdb
        #pdb.set_trace()
        self.env = env
        self.agent_comm = mpi_settings.agent_comm

        # MPI
        self.rank = self.agent_comm.rank
        self.size = self.agent_comm.size

        ## Default settings 
        num_cores = os.cpu_count()
        num_CPU   = os.cpu_count()
        num_GPU   = 0

        ## Setup GPU cfg
        if tf_version < 2:
            gpu_names = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
            if self.rank==0 and len(gpu_names)>0:
                    num_cores = 1
                    num_CPU   = 1
                    num_GPU   = len(gpu_names)
            config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                        inter_op_parallelism_threads=num_cores, 
                        allow_soft_placement=True,
                        device_count = {'CPU' : num_CPU,
                                        'GPU' : num_GPU})
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            set_session(sess)
        elif tf_version >= 2:

            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.compat.v1.Session(config=config)
            tf.compat.v1.keras.backend.set_session(sess)

        ## Declare hyper-parameters, initialized for determining datatype
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
        self.dense = [0, 0]
        self.activation = 'relu'
        self.optimizer = 'adam'
        self.loss = 'mse'

        ## WRONG ASSUMPTION ##
        ## TODO: Assuming rank==0 is the only learner 
        #self.memory = deque(maxlen = 0)
        #if self.rank==0:
        self.memory = deque(maxlen = 1000) ## TODO: make configurable

    def set_agent(self):
        # Get hyper-parameters
        agent_data = super().get_config()

        self.results_dir = agent_data['output_dir']
        self.gamma =  agent_data['gamma']
        self.epsilon = agent_data['epsilon']
        self.epsilon_min = agent_data['epsilon_min']
        self.epsilon_decay = agent_data['epsilon_decay']
        self.learning_rate = agent_data['learning_rate']
        self.batch_size = agent_data['batch_size']
        self.tau = agent_data['tau']
        self.model_type = agent_data['model_type']
        self.dense = agent_data['dense']
        self.activation = agent_data['activation']
        self.optimizer = agent_data['optimizer']

        # Build network model
        print("Model: ")
        self.model = self._build_model()
        print("Target model: ")
        self.target_model = self._build_model()
        self.target_weights = self.target_model.get_weights()

    def _huber_loss(self, target, prediction):
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1,axis=-1)
        
    def _build_model(self):
        if self.model_type == 'MLP':
            from agents.agent_vault._build_mlp import build_model
            return build_model(self)
        elif self.model_type == 'LSTM':
            from agents.agent_vault._build_lstm import build_model
            return build_model(self)
        else:
            sys.exit("Oops! That was not a valid model type. Try again...")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def action(self, state):
        random.seed(datetime.now())
        random_data = os.urandom(4)
        np.random.seed(int.from_bytes(random_data, byteorder="big"))
        rdm = np.random.rand()
        #print('epsilon:',self.epsilon)
        if rdm <= self.epsilon:
            action = random.randrange(self.env.action_space.n)
            return action , 0
        else:
            np_state = np.array(state).reshape(1,1,len(state))
            act_values = self.target_model.predict(np_state)
            action = np.argmax(act_values[0])
            return action , 1

    def play(self,state):
        act_values = self.target_model.predict(state)
        return np.argmax(act_values[0])

    def generate_data(self):
        ''' Worker method to create samples for training '''
        batch_states = []
        batch_target = []
        ## Return empty batch
        if len(self.memory)<self.batch_size:
            yield batch_states, batch_target

        start_time_episode = time.time()
        minibatch = random.sample(self.memory, self.batch_size)
        #logger.info('Agent - Minibatch time: %s ' % (str(time.time() - start_time_episode)))
        for state, action, reward, next_state, done in minibatch:
            np_state = np.array(state).reshape(1, 1, len(state))
            np_next_state = np.array(next_state).reshape(1, 1, len(next_state))
            expectedQ = 0
            if not done:
                expectedQ = self.gamma * np.amax(self.target_model.predict(np_next_state)[0])
            target = reward + expectedQ
            import pdb
            pdb.set_trace()
            target_f = self.target_model.predict(np_state)
            target_f[0][action] = target
            if batch_states == []:
                batch_states = np_state
                batch_target = target_f
            else:
                batch_states = np.append(batch_states, np_state, axis=0)
                batch_target = np.append(batch_target, target_f, axis=0)
        #logger.info('Agent - Data generator time: %s ' % (str(time.time() - start_time_episode)))

        yield batch_states,batch_target


    def train(self, batch):
        self.epsilon_adj()
        batch_states, batch_target = batch

        if len(self.memory) > (self.batch_size) and len(batch_states)>=(self.batch_size):
            start_time_episode = time.time()
            history = self.model.fit(batch_states, batch_target, epochs=2, verbose=2)
            logger.info('Agent[%s]- Training: %s ' % (str(self.rank), str(time.time() - start_time_episode)))
            start_time_episode = time.time()
            logger.info('Agent[%s] - Target update time: %s ' % (str(self.rank), str(time.time() - start_time_episode)))

    def get_weights(self):
        #logger.info('Agent[%s] - get target weight.' % str(self.rank))
        return self.target_model.get_weights()

    def set_weights(self, weights):
        #logger.info('Agent[%s] - set target weight.' % str(self.rank))
        #logger.info('Agent[%s] - set target weight: %s' % (str(self.rank),weights))
        self.target_model.set_weights(weights)
        
    def target_train(self):
        #logger.info('Agent[%s] - update target weights.' % str(self.rank))
        #self.target_train_counter = 0
        model_weights  = self.model.get_weights()
        target_weights =self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = self.tau*model_weights[i] + (1-self.tau)*target_weights[i]
        self.set_weights(target_weights)

    def epsilon_adj(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, filename):
        #self.model.load_weights(filename)
        layers = self.model.layers
        with open(filename, 'rb') as f:
            pickle_list = pickle.load(f)

        for layerId in range(len(layers)):
            assert(layers[layerId].name == pickle_list[layerId][0])
            layers[layerId].set_weights(pickle_list[layerId][1])

    def save(self, filename):
        #self.model.save_weights(filename)
        layers = self.model.layers
        pickle_list = []
        for layerId in range(len(layers)):
            weigths = layers[layerId].get_weights()
            pickle_list.append([layers[layerId].name, weigths])

        with open(filename, 'wb') as f:
            pickle.dump(pickle_list, f, -1)

    def update(self):
        print("Implement update method in dqn.py")

    def monitor(self):
        print("Implement monitor method in dqn.py")

    def benchmark(dataset, num_epochs=1):
        start_time = time.perf_counter()
        for epoch_num in range(num_epochs):
            for sample in dataset:
                # Performing a training step
                time.sleep(0.01)
                print(sample)
        tf.print("Execution time:", time.perf_counter() - start_time)
