import mpi4py.rc; mpi4py.rc.threads = False
from mpi4py import MPI
import random,sys,os
import numpy as np
from collections import deque
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Input,GaussianNoise,BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
import csv,json,math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

import tensorflow as tf
import keras as keras
from keras.backend.tensorflow_backend import set_session

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RL-Logger')
logger.setLevel(logging.ERROR)

#The Deep Q-Network (DQN)
class DQN:
    def __init__(self, env,cfg='cfg/dqn_setup.json'):
        self.env = env
        self.memory = deque(maxlen = 2000)
        self.avg_reward = 0
        self.target_train_counter = 0

        self.total_actions_taken = 1
        self.individual_action_taken = np.ones(self.env.action_space.n)

        ##
        world_comm = MPI.COMM_WORLD
        world_rank = world_comm.rank
        if world_rank==0:
            os.environ["CUDA_VISIBLE_DEVICES"]="0"##,1,2,3"
        else: 
            os.environ["CUDA_VISIBLE_DEVICES"]=""

        tf_version = int((tf.__version__)[0])
        
        if tf_version < 2:
            ## Setup GPU cfg
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            set_session(sess)
        elif tf_version >= 2:
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.compat.v1.Session(config=config)
            tf.compat.v1.keras.backend.set_session(sess)
        
        ## Get hyper-parameters from json cfg file
        data = []
        with open(cfg) as json_file:
            data = json.load(json_file)
            
        self.search_method = "epsilon"
        self.gamma =  float(data['gamma']) if float(data['gamma']) else 0.95  # discount rate
        self.epsilon = float(data['epsilon']) if float(data['epsilon']) else 1.0  # exploration rate
        self.epsilon_min = float(data['epsilon_min']) if float(data['epsilon_min']) else 0.05
        self.epsilon_decay = float(data['epsilon_decay']) if float(data['epsilon_decay']) else 0.995
        self.learning_rate =  float(data['learning_rate']) if float(data['learning_rate']) else  0.001
        self.batch_size = int(data['batch_size']) if int(data['batch_size']) else 32
        self.target_train_interval =  50
        self.tau = float(data['tau']) if float(data['tau']) else 1.0
        self.save_model = './model/'

        self.model = self._build_model()
        self.target_model = self._build_model()

        ## Save infomation ##
        train_file_name = "dqn_huber_clipnorm=1_clipvalue05_online_accelerator_lr%s_v4.log" % str(self.learning_rate) 
        self.train_file = open(train_file_name, 'w')
        self.train_writer = csv.writer(self.train_file, delimiter = " ")

    ##@profile
    def _build_model(self):
        ## Input: state ##       
        state_input = Input(self.env.observation_space.shape)
        ## Make noisy input data ##
        #state_input = GaussianNoise(0.1)(state_input)
        ## Noisy layer 
        h1 = Dense(56, activation='tanh')(state_input)
        h1 = GaussianNoise(0.1)(h1)
        ## Noisy layer
        h2 = Dense(56, activation='tanh')(h1)
        h2 = GaussianNoise(0.1)(h2)
        ## Output layer
        h3 = Dense(56, activation='tanh')(h2)
        ## Output: action ##   
        output = Dense(self.env.action_space.n,activation='linear')(h3)
        model = Model(input=state_input, output=output)
        adam = Adam(lr=self.learning_rate, clipnorm=1.0, clipvalue=0.5) ## clipvalue=0.5,clipnorm=1.0,)
        model.compile(loss='mse', optimizer=adam)
        #model.summary()
        return model       

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def action(self, state):
        if np.random.rand() <= self.epsilon:
            #logger.info('Random action')
            action = random.randrange(self.env.action_space.n)
            ## Update randomness
            if len(self.memory)>(self.batch_size):
                self.epsilon_adj()
        else:
            #logger.info('NN action')
            np_state = np.array(state).reshape(1,len(state))
            act_values = self.target_model.predict(np_state)
            action = np.argmax(act_values[0])

        return action

    def play(self,state):
        act_values = self.target_model.predict(state)
        return np.argmax(act_values[0])

    def train(self):
        if len(self.memory)<(self.batch_size):
            return

        #logger.info('### TRAINING MODEL ###')
        losses = []
        minibatch = random.sample(self.memory, self.batch_size)
        batch_states = []
        batch_target = []
        for state, action, reward, next_state, done in minibatch:
 
            np_state = np.array(state).reshape(1,len(state))
            np_next_state = np.array(next_state).reshape(1,len(next_state))
            expectedQ =0 
            if not done:
                expectedQ = self.gamma*np.amax(self.target_model.predict(np_next_state)[0])
            target = reward + expectedQ
            target_f = self.target_model.predict(np_state)
            target_f[0][action] = target
            
            if batch_states==[]:
                batch_states=np_state
                batch_target=target_f
            else:
                batch_states=np.append(batch_states,np_state,axis=0)
                batch_target=np.append(batch_target,target_f,axis=0)
                
        history = self.model.fit(batch_states, batch_target, epochs = 1, verbose = 0)
        losses.append(history.history['loss'][0])
        self.train_writer.writerow([np.mean(losses)])
        self.train_file.flush()
        
        if self.target_train_counter%self.target_train_interval == 0:
            #logger.info('### TRAINING TARGET MODEL ###')
            self.target_train()
            
        return np.mean(losses)

    def target_train(self):
        self.target_train_counter = 0
        model_weights  = self.model.get_weights()
        target_weights =self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = self.tau*model_weights[i] + (1-self.tau)*target_weights[i]
        self.target_model.set_weights(target_weights)

    def epsilon_adj(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        abspath = os.path.abspath(self.save_model + name)
        path = os.path.dirname(abspath)
        if not os.path.exists(path):os.makedirs(path)
        # Save JSON config to disk
        model_json_name = self.save_model + name + '.json'
        json_config = self.model.to_json()
        with open(model_json_name, 'w') as json_file:
            json_file.write(json_config)
        # Save weights to disk
        self.model.save_weights(self.save_model + name+'.weights.h5')
        self.model.save(self.save_model + name+'.modelall.h5')
        #logger.info('### SAVING MODEL '+abspath+'###')
