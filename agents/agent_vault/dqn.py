import random,sys,os
import numpy as np
from collections import deque
import tensorflow as tf
import csv
import json
import math
import logging
import exarl as erl
import pickle
from keras.backend.tensorflow_backend import set_session
tf_version = int((tf.__version__)[0])

#if tf_version < 2:
#    from tensorflow.keras.models import Sequential,Model
#    from tensorflow.keras.layers import Dense,Dropout,Input,BatchNormalization
#    from tensorflow.keras.optimizers import Adam
#    from tensorflow.keras import backend as K
from tensorflow.python.client import device_lib
#elif tf_version >=2:
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Input,BatchNormalization
from keras.optimizers import Adam
from keras import backend as K

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RL-Logger')
logger.setLevel(logging.INFO)
#logger.setLevel(logging.NOTSET)

#The Deep Q-Network (DQN)
class DQN(erl.ExaAgent):
    def __init__(self, env, agent_comm, cfg='agents/agent_vault/agent_cfg/dqn_setup.json'):

        self.env = env
        self.agent_comm = agent_comm
        self.default_cfg = 'agents/agent_vault/agent_cfg/dqn_setup.json'
        ## Implement the UCB approach
        self.sigma = 2 # confidence level
        self.total_actions_taken = 1
        self.individual_action_taken = np.ones(self.env.action_space.n)

        # MPI
        self.rank = self.agent_comm.rank
        self.size = self.agent_comm.size
        #logger.info("Rank: %s" % self.rank)
        #logger.info("Size: %s" % self.size)

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
        #super().__init__(agent_cfg=cfg)
        self.search_method = ''
        self.gamma = 0
        self.epsilon = 0
        self.epsilon_min = 0
        self.epsilon_decay = 0
        self.learning_rate = 0
        self.batch_size = 0
        self.tau = 0
               
        ## TODO: Assuming rank==0 is the only learner
        self.memory = deque(maxlen = 0)
        if self.rank==0:
            deque(maxlen = 2000) ## TODO: make configurable
            
        ##
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_weights = self.target_model.get_weights()
       
    def stage_info(self):
        # Get results directory
        self.results_dir = super().get_results_dir()
        # Get hyper-parameters                                                                                                                    
        agent_data = super().get_config()
        self.search_method =  (agent_data['search_method'])
        self.gamma =  (agent_data['gamma'])
        self.epsilon = (agent_data['epsilon'])
        self.epsilon_min = (agent_data['epsilon_min'])
        self.epsilon_decay = (agent_data['epsilon_decay'])
        self.learning_rate =  (agent_data['learning_rate'])
        self.batch_size = (agent_data['batch_size'])
        self.tau = (agent_data['tau'])

        train_file_name = "dqn_exacartpole_%s_lr%s_tau%s_v1.log" % (self.search_method, str(self.learning_rate) ,str(self.tau) )
        self.train_file = open(self.results_dir + train_file_name, 'w')
        self.train_writer = csv.writer(self.train_file, delimiter = " ")

    def _huber_loss(self, target, prediction):
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1,axis=-1)

    def _build_model(self):
        ## Input: state ##       
        state_input = Input(self.env.observation_space.shape)
        #s1 = BatchNormalization()(state_input)
        h1 = Dense(64, activation='relu')(state_input)
        #b1 = BatchNormalization()(h1)
        h2 = Dense(128, activation='relu')(h1)
        #b2 = BatchNormalization()(h2)
        #h3 = Dense(24, activation='relu')(h2)
        ## Output: action ##   
        output = Dense(self.env.action_space.n,activation='relu')(h2)
        model = Model(inputs=state_input, outputs=output)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        #model.compile(loss='mean_squared_logarithmic_error', optimizer=adam)
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def action(self, state):
        action = -1
        ## TODO: Update greed-epsilon to something like UBC
        if np.random.rand() <= self.epsilon and self.search_method=="epsilon":
            logger.info('Random action')
            action = random.randrange(self.env.action_space.n)
            ## Update randomness
            #if len(self.memory)>(self.batch_size):
            self.epsilon_adj()

        else:
            logger.info('Policy action')
            np_state = np.array(state).reshape(1,len(state))
            act_values = self.target_model.predict(np_state)
            action = np.argmax(act_values[0])
            mask = [i for i in range(len(act_values[0])) if act_values[0][i] == act_values[0][action]]
            ncands=len(mask)
            # print( 'Number of cands: %s' % str(ncands))
            if ncands>1:
                action = mask[random.randint(0,ncands-1)]
        ## Capture the action statistics for the UBC methods
        #print('total_actions_taken: %s' % self.total_actions_taken)
        #print('individual_action_taken[%s]: %s' % (action,self.individual_action_taken[action]))
        self.total_actions_taken += 1
        self.individual_action_taken[action]+=1

        return action

    def play(self,state):
        act_values = self.target_model.predict(state)
        return np.argmax(act_values[0])

    def train(self):
        if len(self.memory)<(self.batch_size):
            return
        logger.info('### TRAINING ###')
        losses = []
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            np_state = np.array(state).reshape(1,len(state))
            np_next_state = np.array(next_state).reshape(1,len(next_state))
            expectedQ =0
            if not done:
                expectedQ = self.gamma*np.amax(self.target_model.predict(np_next_state)[0])
            target = reward + expectedQ
            target_f = self.model.predict(np_state)
            target_f[0][action] = target
            history = self.model.fit(np_state, target_f, epochs = 1, verbose = 0)
            losses.append(history.history['loss'])
        self.target_train()
        self.train_writer.writerow([np.mean(losses)])
        self.train_file.flush()

    def target_train(self):
        if len(self.memory)%(self.batch_size)!=0:
            return
        logger.info('### TARGET UPDATE ###')
        model_weights  = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = self.tau*model_weights[i] + (1-self.tau)*target_weights[i]
        self.target_model.set_weights(target_weights)
        self.target_weights = target_weights

    def epsilon_adj(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            logger.info('New epsilon: %s ' % self.epsilon)

    def get_weights(self):
        return self.target_model.get_weights()

    def set_weights(self, weights):
        self.target_model.set_weights(weights)
    
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
