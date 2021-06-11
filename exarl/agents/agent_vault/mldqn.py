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
import gym
import pickle
import exarl as erl
from exarl.base.comm_base import ExaComm
from tensorflow import keras
from collections import deque
from datetime import datetime
import numpy as np
from exarl.agents.agent_vault._prioritized_replay import PrioritizedReplayBuffer
import exarl.utils.candleDriver as cd
import exarl.utils.log as log
from exarl.utils.introspect import introspectTrace
from tensorflow.compat.v1.keras.backend import set_session
try:
    import horovod.tensorflow as hvd
    horovod_imported = False
except:
    horovod_imported = False

logger = log.setup_logger(__name__, cd.run_params['log_level'])

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.loss = []

    def on_batch_end(self, batch, logs={}):
        self.loss.append(logs.get('loss'))

# The Discrete Double Deep Q-Network (DDDQN)
class MLDQN(erl.ExaAgent):
    def __init__(self, env, is_learner):
        # Initial values
        self.is_learner = is_learner
        self.model = None
        self.target_model = None
        self.target_weights = None
        self.device = None
        self.mirrored_strategy = None

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

        # Optimization using XLA (1.1x speedup)
        # tf.config.optimizer.set_jit(True)

        # Optimization using mixed precision (1.5x speedup)
        # Layers use float16 computations and float32 variables
        # from tensorflow.keras.mixed_precision import experimental as mixed_precision
        # policy = mixed_precision.Policy('mixed_float16')
        # git diff
        # mixed_precision.set_policy(policy)

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

        if self.model_type == 'MLP':
            # for mlp
            self.dense = cd.run_params['dense']

        if self.model_type == 'LSTM':
            # for lstm
            self.lstm_layers = cd.run_params['lstm_layers']
            self.gauss_noise = cd.run_params['gauss_noise']
            self.regularizer = cd.run_params['regularizer']
            self.clipnorm = cd.run_params['clipnorm']
            self.clipvalue = cd.run_params['clipvalue']

        # for both
        self.activation = cd.run_params['activation']
        self.out_activation = cd.run_params['out_activation']
        self.optimizer = cd.run_params['optimizer']
        self.loss = cd.run_params['loss']
        self.n_actions = cd.run_params['nactions']
        self.priority_scale = cd.run_params['priority_scale']

        self.is_discrete = (type(env.action_space) == gym.spaces.discrete.Discrete)
        if not self.is_discrete:
            env.action_space.n = self.n_actions
            self.actions = np.linspace(env.action_space.low, env.action_space.high, self.n_actions)

        # Default settings
        num_cores = os.cpu_count()
        num_CPU = os.cpu_count()
        num_GPU = 0

        # Setup GPU cfg
        if self.rank == 0:
            print("Setting GPU rank", self.rank)
            config = tf.compat.v1.ConfigProto(device_count={'GPU': 1, 'CPU': 1})
        else:
            print("Setting no GPU rank", self.rank)
            config = tf.compat.v1.ConfigProto(device_count={'GPU': 0, 'CPU': 1})

        cpus = tf.config.experimental.list_physical_devices("CPU")
        logger.info("Available CPUs: {}".format(cpus))

        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(sess)

        # Build network model
        if self.is_learner:
            with tf.device(self.device):
                self.model = self._build_model()
                self.model.compile(loss=self.loss, optimizer=self.optimizer)
                self.model.summary()
            # self.mirrored_strategy = tf.distribute.MirroredStrategy()
            # logger.info("Using learner strategy: {}".format(self.mirrored_strategy))
            # with self.mirrored_strategy.scope():
            #     self.model = self._build_model()
            #     self.model._name = "learner"
            #     self.model.compile(loss=self.loss, optimizer=self.optimizer)
            #     logger.info("Active model: \n".format(self.model.summary()))
        else:
            self.model = None
        with tf.device('/CPU:0'):
            self.target_model = self._build_model()
            self.target_model._name = "target_model"
            self.target_model.compile(loss=self.loss, optimizer=self.optimizer)
            # self.target_model.summary()
            self.target_weights = self.target_model.get_weights()

        if horovod_imported and ExaComm.is_learner():
            hvd.init(comm=ExaComm.learner_comm.raw())
            self.first_batch = 1
            self.loss_fn = tf.losses.MeanSquaredError()
            self.opt = tf.optimizers.Adam(self.learning_rate * hvd.size())

        self.maxlen = cd.run_params['mem_length']
        self.replay_buffer = PrioritizedReplayBuffer(maxlen=self.maxlen)

    def _get_device(self):
        cpus = tf.config.experimental.list_physical_devices('CPU')
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
            from exarl.agents.agent_vault._build_mlp import build_model
            return build_model(self)
        elif self.model_type == 'LSTM':
            from exarl.agents.agent_vault._build_lstm import build_model
            return build_model(self)
        else:
            sys.exit("Oops! That was not a valid model type. Try again...")

    def set_learner(self):
        logger.debug(
            "Agent[{}] - Creating active model for the learner".format(self.rank)
        )

    def remember(self, state, action, reward, next_state, done):
        lost_data = self.replay_buffer.add((state, action, reward, next_state, done))
        if lost_data and self.priority_scale:
            logger.warning("Priority replay buffer size too small. Data loss negates replay effect!")

    def get_action(self, state):
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

    @introspectTrace()
    def action(self, state):
        action, policy = self.get_action(state)
        if not self.is_discrete:
            action = [self.actions[action]]
        return action, policy

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
        action_idx = action if self.is_discrete else np.where(self.actions == action)[1]
        target_f[0][action_idx] = target
        return target_f[0]

    @introspectTrace()
    def generate_data(self):
        # Worker method to create samples for training
        batch_states = np.zeros((self.batch_size, 1, self.env.observation_space.shape[0]))
        batch_target = np.zeros((self.batch_size, self.env.action_space.n))
        indices = -1 * np.ones(self.batch_size)
        importance = np.ones(self.batch_size)
        if self.replay_buffer.get_buffer_length() < self.batch_size:
            yield batch_states, batch_target, indices, importance
            # if self.priority_scale > 0:
            #     yield batch_states, batch_target, indices, importance
            # else:
            #     print("INIT", len(batch_states), len(batch_target))
            #     yield batch_states, batch_target

        start_time = time.time()
        minibatch, importance, indices = self.replay_buffer.sample(self.batch_size, priority_scale=self.priority_scale)
        batch_target = list(map(self.calc_target_f, minibatch))
        batch_states = [np.array(exp[0]).reshape(1, 1, len(exp[0]))[0] for exp in minibatch]
        batch_states = np.reshape(batch_states, [len(minibatch), 1, len(minibatch[0][0])])
        batch_target = np.reshape(batch_target, [len(minibatch), self.env.action_space.n])
        end_time = time.time()
        self.dataprep_time += (end_time - start_time)
        self.ndataprep_time += 1
        logger.debug('Agent[{}] - Minibatch time: {} '.format(self.rank, (end_time - start_time)))
        yield batch_states, batch_target, indices, importance
        # TODO: THIS BREAKS THE PICKLE
        # if self.priority_scale > 0:
        #     yield batch_states, batch_target, indices, importance
        # else:
        #     print("GENERATING", len(batch_states), len(batch_target))
        #     yield batch_states, batch_target

    def train(self, batch):
        if self.is_learner:
            if batch[2][0] != -1:
                start_time = time.time()
                with tf.device(self.device):
                    # print("Importance = ", sample_weight)
                    if horovod_imported:
                        loss = self.training_step(batch)
                    else:
                        loss = LossHistory()
                        sample_weight = batch[3] * (1 - self.epsilon)
                        self.model.fit(batch[0], batch[1], epochs=1, batch_size=1, verbose=0, callbacks=loss, sample_weight=sample_weight)
                        loss = loss.loss
                end_time = time.time()
                self.training_time += (end_time - start_time)
                self.ntraining_time += 1
                logger.info('Agent[{}]- Training: {} '.format(self.rank, (end_time - start_time)))
                start_time_episode = time.time()
                logger.info('Agent[%s] - Target update time: %s ' % (str(self.rank), str(time.time() - start_time_episode)))
                return batch[2], loss
        else:
            logger.warning('Training will not be done because this instance is not set to learn.')

        return -1 * np.ones(self.batch_size), -1 * np.ones(self.batch_size)

    # @tf.function
    def training_step(self, batch):
        with tf.GradientTape() as tape:
            probs = self.model(batch[0], training=True)
            loss_value = tf.reduce_mean(tf.multiply(tf.square(self.loss_fn(batch[1], probs)), batch[3] * (1 - self.epsilon)))
            print("LOSS VALUE:", loss_value)

        # Horovod distributed gradient tape
        tape = hvd.DistributedGradientTape(tape)
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

        if self.first_batch:
            hvd.broadcast_variables(self.model.variables, root_rank=0)
            hvd.broadcast_variables(self.opt.variables(), root_rank=0)
            self.first_batch = 0

        return loss_value

    def set_priorities(self, indicies, loss):
        self.replay_buffer.set_priorities(indicies, loss)

    def get_weights(self):
        logger.debug("Agent[%s] - get target weight." % str(self.rank))
        return self.target_model.get_weights()

    def set_weights(self, weights):
        logger.info("Agent[%s] - set target weight." % str(self.rank))
        logger.debug("Agent[%s] - set target weight: %s" % (str(self.rank), weights))
        with tf.device(self.device):
            self.target_model.set_weights(weights)

    def target_train(self):
        if self.is_learner:
            logger.info("Agent[%s] - update target weights." % str(self.rank))
            with tf.device(self.device):
                model_weights = self.model.get_weights()
                target_weights = self.target_model.get_weights()
            for i in range(len(target_weights)):
                target_weights[i] = (
                    self.tau * model_weights[i] + (1 - self.tau) * target_weights[i]
                )
            self.set_weights(target_weights)
        else:
            logger.warning(
                "Weights will not be updated because this instance is not set to learn."
            )

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

    def print_timers(self):
        if self.ntraining_time > 0:
            logger.info(
                "Agent[{}] - Average training time: {}".format(
                    self.rank, self.training_time / self.ntraining_time
                )
            )
        else:
            logger.info("Agent[{}] - Average training time: {}".format(self.rank, 0))

        if self.ndataprep_time > 0:
            logger.info(
                "Agent[{}] - Average data prep time: {}".format(
                    self.rank, self.dataprep_time / self.ndataprep_time
                )
            )
        else:
            logger.info("Agent[{}] - Average data prep time: {}".format(self.rank, 0))
