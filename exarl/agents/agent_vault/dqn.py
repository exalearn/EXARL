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
import os
import sys
import random
import numpy as np
from datetime import datetime

import gym
from gym.spaces.utils import flatten
import tensorflow as tf
from tensorflow import keras

import exarl as erl
from exarl.utils.globals import ExaGlobals
from exarl.base.comm_base import ExaComm
from exarl.candlelib import candle
from exarl.agents.agent_vault._prioritized_replay import PrioritizedReplayBuffer
from exarl.utils.introspect import introspectTrace
logger = ExaGlobals.setup_logger(__name__)

# TODO: ExaComm is probably not set at import time
if ExaComm.num_learners > 1:
    import horovod.tensorflow as hvd
    multiLearner = True
else:
    multiLearner = False

class LossHistory(keras.callbacks.Callback):
    """Loss history for training
    """

    def on_train_begin(self, logs={}):
        self.loss = []

    def on_batch_end(self, batch, logs={}):
        self.loss.append(logs.get('loss'))

# The Multi-Learner Discrete Double Deep Q-Network
class DQN(erl.ExaAgent):
    """Multi-Learner Discrete Double Deep Q-Network with Prioritized Experience Replay
    """

    def __init__(self, env, is_learner):
        """DQN Constructor

        Args:
            env (OpenAI Gym environment object): env object indicates the RL environment
            is_learner (bool): Used to indicate if the agent is a learner or an actor
        """

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

        # Timers
        self.training_time = 0
        self.ntraining_time = 0
        self.dataprep_time = 0
        self.ndataprep_time = 0

        self.enable_xla = True if ExaGlobals.lookup_params('xla') == "True" else False
        if self.enable_xla:
            # Optimization using XLA (1.1x speedup)
            tf.config.optimizer.set_jit(True)

            # Optimization using mixed precision (1.5x speedup)
            # Layers use float16 computations and float32 variables
            from tensorflow.keras.mixed_precision import experimental as mixed_precision
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_policy(policy)

        # dqn intrinsic variables
        self.results_dir = ExaGlobals.lookup_params('output_dir')
        self.gamma = ExaGlobals.lookup_params('gamma')
        self.epsilon = ExaGlobals.lookup_params('epsilon')
        self.epsilon_min = ExaGlobals.lookup_params('epsilon_min')
        self.epsilon_decay = ExaGlobals.lookup_params('epsilon_decay')
        self.learning_rate = ExaGlobals.lookup_params('learning_rate')
        self.batch_size = ExaGlobals.lookup_params('batch_size')
        self.tau = ExaGlobals.lookup_params('tau')
        self.model_type = ExaGlobals.lookup_params('model_type')

        if self.model_type == 'MLP':
            # for mlp
            self.dense = ExaGlobals.lookup_params('dense')

        if self.model_type == 'LSTM':
            # for lstm
            self.lstm_layers = ExaGlobals.lookup_params('lstm_layers')
            self.gauss_noise = ExaGlobals.lookup_params('gauss_noise')
            self.regularizer = ExaGlobals.lookup_params('regularizer')
            self.clipnorm = ExaGlobals.lookup_params('clipnorm')
            self.clipvalue = ExaGlobals.lookup_params('clipvalue')

        # for both
        self.activation = ExaGlobals.lookup_params('activation')
        self.out_activation = ExaGlobals.lookup_params('out_activation')
        self.optimizer = ExaGlobals.lookup_params('optimizer')
        self.loss = ExaGlobals.lookup_params('loss')
        self.n_actions = ExaGlobals.lookup_params('nactions')
        self.priority_scale = ExaGlobals.lookup_params('priority_scale')

        # Check if the action space is discrete
        self.is_discrete = (type(env.action_space) == gym.spaces.discrete.Discrete)
        # If continuous, discretize the action space
        # TODO: Incorpoorate Ai's class
        if not self.is_discrete:
            env.action_space.n = self.n_actions
            self.actions = np.linspace(env.action_space.low, env.action_space.high, self.n_actions)

        # Data types of action and observation space
        self.dim_action = np.array(self.env.action_space.sample()).dtype

        flat_sample = flatten(self.env.observation_space, self.env.observation_space.sample())
        self.dtype_observation = flat_sample.dtype
        self.dim_observation = flat_sample.shape[0]

        # Setup GPU cfg
        if ExaComm.is_learner():
            logger().info("Setting GPU rank", self.rank)
            config = tf.compat.v1.ConfigProto(device_count={'GPU': 1, 'CPU': 1})
        else:
            logger().info("Setting no GPU rank", self.rank)
            config = tf.compat.v1.ConfigProto(device_count={'GPU': 0, 'CPU': 1})
        # Get which device to run on
        self.device = self._get_device()

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
            # logger().info("Using learner strategy: {}".format(self.mirrored_strategy))
            # with self.mirrored_strategy.scope():
            #     self.model = self._build_model()
            #     self.model._name = "learner"
            #     self.model.compile(loss=self.loss, optimizer=self.optimizer)
            #     logger().info("Active model: \n".format(self.model.summary()))
        else:
            self.model = None
        with tf.device('/CPU:0'):
            self.target_model = self._build_model()
            self.target_model._name = "target_model"
            self.target_model.compile(loss=self.loss, optimizer=self.optimizer)
            # self.target_model.summary()
            self.target_weights = self.target_model.get_weights()

        if multiLearner and ExaComm.is_learner():
            hvd.init(comm=ExaComm.learner_comm.raw())
            self.first_batch = 1
            # TODO: Update candle driver to include different losses and optimizers
            # Default reduction is tf.keras.losses.Reduction.AUTO which errors out with distributed training
            # self.loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
            self.loss_fn = candle.build_loss(self.loss, ExaGlobals.keras_defaults(), reduction='none')
            # self.opt = tf.keras.optimizers.Adam(self.learning_rate * hvd.size())
            self.opt = candle.build_optimizer(self.optimizer, self.learning_rate * hvd.size(), ExaGlobals.keras_default())

        self.buffer_capacity = ExaGlobals.lookup_params('buffer_capacity')
        self.replay_buffer = PrioritizedReplayBuffer(maxlen=self.buffer_capacity)

    def _get_device(self):
        """Get device type (CPU/GPU)

        Returns:
            string: device type
        """
        cpus = tf.config.experimental.list_physical_devices('CPU')
        gpus = tf.config.experimental.list_physical_devices('GPU')
        ngpus = len(gpus)
        logger().info('Number of available GPUs: {}'.format(ngpus))
        if ngpus > 0:
            gpu_id = self.rank % ngpus
            return '/GPU:{}'.format(gpu_id)
        else:
            return '/CPU:0'

    def _build_model(self):
        """Build NN model based on parameters provided in the config file

        Returns:
            [type]: [description]
        """
        if self.model_type == 'MLP':
            from exarl.agents.agent_vault._build_mlp import build_model
            return build_model(self)
        elif self.model_type == 'LSTM':
            from exarl.agents.agent_vault._build_lstm import build_model
            return build_model(self)
        else:
            sys.exit("Oops! That was not a valid model type. Try again...")

    def flatten_observation(self, state):
        state = flatten(self.env.observation_space, state)
        if self.model_type == 'LSTM':
            return state.reshape(1, 1, state.shape[0])
        return state.reshape(1, state.shape[0])

    def remember(self, state, action, reward, next_state, done):
        """Add experience to replay buffer

        Args:
            state (list or array): Current state of the system
            action (list or array): Action to take
            reward (list or array): Environment reward
            next_state (list or array): Next state of the system
            done (bool): Indicates episode completion
        """
        lost_data = self.replay_buffer.add((state, action, reward, next_state, done))
        if lost_data and self.priority_scale:
            # logger().warning("Priority replay buffer size too small. Data loss negates replay effect!")
            print("Priority replay buffer size too small. Data loss negates replay effect!", flush=True)

    def get_action(self, state):
        """Use epsilon-greedy approach to generate actions

        Args:
            state (list or array): Current state of the system

        Returns:
            (list or array): Action to take
        """
        random.seed(datetime.now())
        random_data = os.urandom(4)
        np.random.seed(int.from_bytes(random_data, byteorder="big"))
        rdm = np.random.rand()
        if rdm <= self.epsilon:
            self.epsilon_adj()
            action = self.env.action_space.sample()
            return action, 0
        else:
            np_state = self.flatten_observation(state)
            with tf.device(self.device):
                act_values = self.target_model.predict_on_batch(np_state)
            action = np.argmax(act_values[0])
            return action, 1

    @introspectTrace()
    def action(self, state):
        """Discretizes 1D continuous actions to work with DQN

        Args:
            state (list or array): Current state of the system

        Returns:
            action (list or array): Action to take
            policy (int): random (0) or inference (1)
        """
        action, policy = self.get_action(state)

        if not self.is_discrete:
            action = [self.actions[action]]
        return action, policy

    @introspectTrace()
    def calc_target_f(self, exp):
        """Bellman equation calculations

        Args:
            exp (list of experience): contains state, action, reward, next state, done

        Returns:
            target Q value (array): [description]
        """
        state, action, reward, next_state, done = exp
        np_state = self.flatten_observation(state)
        np_next_state = self.flatten_observation(next_state)
        reward = tf.cast(reward, tf.float32)
        np_state = tf.convert_to_tensor(np_state)
        np_next_state = tf.convert_to_tensor(np_next_state)

        expectedQ = 0
        with tf.device(self.device):
            if not done:
                expectedQ = tf.multiply(self.gamma, tf.reduce_max(self.target_model.predict_on_batch(np_next_state)))
                expectedQ = tf.cast(expectedQ, tf.float32)
            target = tf.add(reward, expectedQ)
            target_f = self.target_model.predict_on_batch(np_state)
        # For handling continuous to discrete actions
        action_idx = action if self.is_discrete else np.where(self.actions == action)[1]
        target_f[0][action_idx] = target
        return target_f[0]

    def has_data(self):
        """Indicates if the buffer has data of size batch_size or more

        Returns:
            bool: True if replay_buffer length >= self.batch_size
        """
        # return (self.replay_buffer.get_buffer_length() >= self.batch_size)
        return (self.replay_buffer.get_buffer_length() > 0)

    @introspectTrace()
    def generate_data(self):
        """
        Unpack and yield training data
        Has data checks if the buffer is greater than batch size for training

        Yields:
            batch_states (numpy array): training input
            batch_target (numpy array): training labels
            With PER:
                indices (numpy array): data indices
                importance (numpy array): importance weights
        """
        size = self.batch_size
        if not self.has_data():
            # Worker method to create samples for training
            batch_states = np.zeros((self.batch_size, np.prod(self.dim_observation)), dtype=self.dtype_observation)
            batch_target = np.zeros((self.batch_size, self.env.action_space.n), dtype=self.dim_action)
            indices = -1 * np.ones(self.batch_size)
            importance = np.ones(self.batch_size)
        else:
            minibatch, importance, indices = self.replay_buffer.sample(self.batch_size, priority_scale=self.priority_scale)
            batch_target = list(map(self.calc_target_f, minibatch))
            batch_states = [self.flatten_observation(exp[0]) for exp in minibatch]
            size = len(minibatch)

        # JS: Always reshape... Even the zeros.
        batch_target = np.reshape(batch_target, [size, self.env.action_space.n])
        if self.model_type == 'LSTM':
            batch_states = np.reshape(batch_states, [size, 1, self.dim_observation])
        else:
            batch_states = np.reshape(batch_states, [size, self.dim_observation])

        if self.priority_scale > 0:
            yield batch_states, batch_target, indices, importance
        else:
            yield batch_states, batch_target

    @introspectTrace()
    def train(self, batch):
        """Train the NN

        Args:
            batch (list): sampled batch of experiences

        Returns:
            if PER:
                indices (numpy array): data indices
                loss: training loss
            else:
                None
        """
        ret = None
        with tf.device(self.device):
            if self.priority_scale > 0:
                if multiLearner:
                    loss = self.training_step(batch)
                else:
                    loss = LossHistory()
                    sample_weight = batch[3] ** (1 - self.epsilon)
                    self.model.fit(batch[0], batch[1], epochs=1, batch_size=1, verbose=0, callbacks=loss, sample_weight=sample_weight)
                    loss = loss.loss
                ret = batch[2], loss
            else:
                if multiLearner:
                    loss = self.training_step(batch)
                else:
                    self.model.fit(batch[0], batch[1], epochs=1, verbose=0)
        self.ntraining_time += 1
        return ret

    @tf.function
    def training_step(self, batch):
        """ Training step for multi-learner using Horovod

        Args:
            batch (list): sampled batch of experiences

        Returns:
            loss_value: loss value per training step for multi-learner
        """
        with tf.GradientTape() as tape:
            probs = self.model(batch[0], training=True)
            if len(batch) > 2:
                sample_weight = batch[3] * (1 - self.epsilon)
            else:
                sample_weight = np.ones(len(batch[0]))
            loss_value = self.loss_fn(batch[1], probs, sample_weight=sample_weight)

        # Horovod distributed gradient tape
        tape = hvd.DistributedGradientTape(tape)
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

        if self.first_batch:
            hvd.broadcast_variables(self.model.variables, root_rank=0)
            hvd.broadcast_variables(self.opt.variables(), root_rank=0)
            self.first_batch = 0
        return loss_value

    def set_priorities(self, indices, loss):
        """ Set priorities for training data

        Args:
            indices (array): data indices
            loss (array): Losses
        """
        self.replay_buffer.set_priorities(indices, loss)

    def get_weights(self):
        """Get weights from target model

        Returns:
            weights (list): target model weights
        """
        logger().debug("Agent[%s] - get target weight." % str(self.rank))
        return self.target_model.get_weights()

    def set_weights(self, weights):
        """Set model weights

        Args:
            weights (list): model weights
        """
        logger().info("Agent[%s] - set target weight." % str(self.rank))
        logger().debug("Agent[%s] - set target weight: %s" % (str(self.rank), weights))
        with tf.device(self.device):
            self.target_model.set_weights(weights)

    @introspectTrace()
    def update_target(self):
        """Update target model
        """
        if self.is_learner:
            logger().info("Agent[%s] - update target weights." % str(self.rank))
            with tf.device(self.device):
                model_weights = self.model.get_weights()
                target_weights = self.target_model.get_weights()
            for i in range(len(target_weights)):
                target_weights[i] = (
                    self.tau * model_weights[i] + (1 - self.tau) * target_weights[i]
                )
            self.set_weights(target_weights)
        else:
            logger().warning(
                "Weights will not be updated because this instance is not set to learn."
            )

    def epsilon_adj(self):
        """Update epsilon value
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
