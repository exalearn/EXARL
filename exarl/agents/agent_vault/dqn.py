import numpy as np
import tensorflow as tf
from gym import spaces
from gym.spaces.utils import flatdim
from copy import deepcopy

import exarl
from exarl.utils.globals import ExaGlobals
from exarl.agents.replay_buffers.buffer import Buffer
from exarl.agents.models.tf_model import Tensorflow_Model
from exarl.utils.introspect import introspectTrace

logger = ExaGlobals.setup_logger(__name__)

class DQN(exarl.ExaAgent):

    def __init__(self, env, is_learner):
        self.env = env
        self.is_learner = is_learner
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        assert isinstance(self.action_space, spaces.Discrete), "This agent only supports Discrete: " + type(self.action_space)

        self._num_actions = self.action_space.n
        self._discount = ExaGlobals.lookup_params('discount') #0.99
        self._batch_size = ExaGlobals.lookup_params('batch_size') #32
        self._sgd_period = ExaGlobals.lookup_params('sgd_period') #1
        self._target_update_period = ExaGlobals.lookup_params('update_target_frequency') #4
        self._epsilon = ExaGlobals.lookup_params('epsilon') #0.05
        self._min_replay_size = ExaGlobals.lookup_params('min_replay_size') #100
        seed = ExaGlobals.lookup_params('seed')
        self._replay = Buffer.create(observation_space=self.observation_space, action_space=self.action_space)

        # JS: We will get fake data for sonnet model and RMA
        fake_data = self._replay.get_fake_data(self._batch_size)
        fake_data = (self._prep_data(fake_data), -1)

        # JS: Seed tf RNG
        tf.random.set_seed(seed)
        self._rng = np.random.RandomState(seed)
        
        self._steps_since_generate_data = 0
        self._step_since_update = 0

        self._init_model(fake_data)

        # JS: This allows for RMA windows to be set up
        self.rma_model = self.get_weights()
        # JS: This is setup for Priority Experience Replay
        self.rma_train_ret = (np.zeros(shape=(self._batch_size,), dtype=np.float32), [0] * self._batch_size)
        self.rma_exp_data = fake_data

    def _init_model(self, fake_data):
        network = Tensorflow_Model.create(observation_space=self.observation_space, 
                                   action_space=self.action_space, 
                                   use_gpu=self.is_learner)
        self._online_network = network
        
        strategy = self._online_network.model.optimizer._distribution_strategy
        self._online_network.model.optimizer._distribution_strategy = None
        self._target_network = deepcopy(network)
        self._online_network.model.optimizer._distribution_strategy = strategy
        self._target_network.model.optimizer._distribution_strategy = strategy

        self._forward = tf.function(network)
        # JS: This will build the network
        self._online_network.model
        self._target_network.model
        self._optimizer = self._online_network.model.optimizer
        # JS: We do this to match sonnet...
        self._optimizer.apply = lambda x, y: self._optimizer.apply_gradients(zip(x, y))
        self._forward = tf.function(network)

    def get_weights(self):
        """
        Get weights from target model

        Returns
        -------
            list : 
                Target model weights
        """
        return self._online_network.get_weights(), self._target_network.get_weights()

    def set_weights(self, weights):
        """
        Set model weights

        Parameters
        ----------
            weights : list
                Model weights
        """
        online_weights, target_weights = weights
        self._online_network.set_weights(online_weights)
        self._target_network.set_weights(target_weights)

    def action(self, state):
        # JS: Epsilon-greedy policy.
        if self._rng.rand() < self._epsilon:
            action = self.action_space.sample()
            return action, 0
        print(state, flush=True)
        observation = tf.convert_to_tensor(state[None, ...])
        q_values = self._forward(observation)

        q_values = q_values.numpy()
        action = self._rng.choice(np.flatnonzero(q_values == q_values.max()))
        return int(action), 0

    def remember(self, state, action, reward, next_state, done):
        # JS: discount = 0.0 if done else 1.0
        self._replay.store(state, action, reward, next_state, done)
        self._steps_since_generate_data += 1

    def has_data(self):
        return self._replay.size >= self._min_replay_size

    def _prep_data(self, data):
        # JS: This is for the train step which requires "discount"
        data[4] = np.logical_not(data[4]).astype(float)
        return data
    
    def generate_data(self):
        if self.has_data():
            data = self._replay.sample(self._batch_size)
            ret = (self._prep_data(data), self._steps_since_generate_data)
            self._steps_since_generate_data = 0
        else:
            ret = (None, -1)
        yield ret

    def train(self, batch):
        data, steps = batch
        # JS: we set the steps to -1 for fake data
        if steps > 0:
            update = False
            self._step_since_update += steps
            if self._step_since_update >= self._target_update_period:
                update = True
                self._step_since_update = 0
            td_error = self._training_step(data[:5], update)
            # JS: We are using prioritized replay
            if len(data) == 6:
                return td_error.numpy(), data[5]

    @tf.function
    def _training_step(self, data, update):
        # JS: Consider where we prep data
        observations, actions, rewards, next_observations, discounts = data
        rewards = tf.cast(rewards, tf.float32)
        discounts = tf.cast(discounts, tf.float32)
        observations = tf.convert_to_tensor(observations)
        next_observations = tf.convert_to_tensor(next_observations)

        with tf.GradientTape() as tape:
            q_tm1 = self._online_network(observations)
            q_t = self._target_network(next_observations)

            onehot_actions = tf.one_hot(actions, depth=self._num_actions)
            qa_tm1 = tf.reduce_sum(q_tm1 * onehot_actions, axis=-1)
            qa_t = tf.reduce_max(q_t, axis=-1)

            # One-step Q-learning loss.
            target = rewards + discounts * self._discount * qa_t
            td_error = qa_tm1 - target
            loss = 0.5 * tf.reduce_mean(td_error ** 2)

        # Update the online network via SGD.
        variables = self._online_network.trainable_variables
        tape = Horovod_Model.gradient_tape(tape)
        gradients = tape.gradient(loss, variables)
        self._optimizer.apply(gradients, variables)

        # Update target network
        if update:
            for target, param in zip(self._target_network.trainable_variables, self._online_network.trainable_variables):
                target.assign(param)
        return td_error

    def train_return(self, args):
        # JS: This will only be called with Prioratized Replay
        self._replay.update_priorities(*args)

class DQN_v1(DQN):

    def __init__(self, env, is_learner):
        assert ExaGlobals.lookup_params('workflow') != 'sync'
        super(DQN_v1, self).__init__(env, is_learner)
        # JS: This gets the max size of the simple buffer
        batch_episode_frequency = ExaGlobals.lookup_params('batch_episode_frequency')
        batch_step_frequency = ExaGlobals.lookup_params('batch_step_frequency')
        steps = ExaGlobals.lookup_params('n_steps')
        if batch_episode_frequency > 1:
            capacity = batch_episode_frequency * steps
        else:
            if batch_step_frequency == -1:
                batch_step_frequency = steps
            capacity = batch_episode_frequency

        if not is_learner:
            # JS: We create simple buffer for non-learners.
            # They will just send all exps each time to the learner
            self._replay = Buffer.create("SimpleBuffer", capacity=capacity, observation_space=self.observation_space, action_space=self.action_space)

        # JS: This fake data should convert bool to float
        fake_data = self._replay.get_fake_data(capacity)
        self.rma_exp_data = fake_data
        
    def has_data(self):
        return self._replay.size > 0

    def generate_data(self):
        if self.has_data():
            data = self._replay.sample(self._batch_size)
            ret = (data, self._steps_since_generate_data)
            self._steps_since_generate_data = 0
        else:
            ret = (None, -1)
        yield ret

    def train(self, batch):
        data, steps = batch
        # JS: we set the steps to -1 for fake data
        if steps > 0:
            # JS: Store data to the central buffer
            self._replay.bulk_store(data)
            if self._replay.size >= self._min_replay_size:
                # JS: Now we sample and reformat discount
                data = self._replay.sample(self._batch_size)
                batch = (self._prep_data(data), steps)
                # JS: Now we just call super!
                super().train(batch)

class DQN_v2(DQN):

    def __init__(self, env, is_learner):
        self._batch_size = ExaGlobals.lookup_params('batch_size')
        self._trajectory_length = ExaGlobals.lookup_params('trajectory_length')
        assert ExaGlobals.lookup_params('model_type') == 'LSTM'
        assert ExaGlobals.lookup_params('buffer') == 'TrajectoryBuffer'
        assert ExaGlobals.lookup_params('buffer_trajectory_length') == self._trajectory_length
        self._dims = (self._batch_size, self._trajectory_length, flatdim(env.observation_space))
        super(DQN_v2, self).__init__(env, is_learner)
        self._local = Buffer.create(capacity=self._trajectory_length, observation_space=self.observation_space, action_space=self.action_space)
    
    def _prep_data(self, data):
        data[0] = data[0].reshape(self._dims)
        data[1] = data[1][self._dims[1]-1::self._dims[1]]
        data[2] = data[2][self._dims[1]-1::self._dims[1]]
        data[3] = data[3].reshape(self._dims)
        data[4] = data[4][self._dims[1]-1::self._dims[1]]
        data[4] = np.logical_not(data[4]).astype(float)
        return data

    def remember(self, state, action, reward, next_state, done):
        self._local.store(state, action, reward, next_state, done)
        super().remember(state, action, reward, next_state, done)

    def action(self, state):
        # JS: Epsilon-greedy policy.
        if self._rng.rand() < self._epsilon:
            action = self.action_space.sample()
            return action, 0

        # JS: This is the part that is different!
        observation = self._local.last()[0][None, ...]

        observation = tf.convert_to_tensor(observation)
        q_values = self._forward(observation)

        q_values = q_values.numpy()
        action = self._rng.choice(np.flatnonzero(q_values == q_values.max()))
        return int(action), 0

class DQN_v3(DQN):
    def __init__(self, env, is_learner):
        assert ExaGlobals.lookup_params('workflow') != 'rma'
        from exarl.agents.models.hvd_model import Horovod_Model
        super(DQN_v3, self).__init__(env, is_learner)
        Horovod_Model.init_horovod_learners()

    @tf.function
    def _training_step(self, data, update):
        # JS: Consider where we prep data
        observations, actions, rewards, next_observations, discounts = data
        rewards = tf.cast(rewards, tf.float32)
        discounts = tf.cast(discounts, tf.float32)
        observations = tf.convert_to_tensor(observations)
        next_observations = tf.convert_to_tensor(next_observations)

        with tf.GradientTape() as tape:
            q_tm1 = self._online_network(observations)
            q_t = self._target_network(next_observations)

            onehot_actions = tf.one_hot(actions, depth=self._num_actions)
            qa_tm1 = tf.reduce_sum(q_tm1 * onehot_actions, axis=-1)
            qa_t = tf.reduce_max(q_t, axis=-1)

            # One-step Q-learning loss.
            target = rewards + discounts * self._discount * qa_t
            td_error = qa_tm1 - target
            loss = 0.5 * tf.reduce_mean(td_error ** 2)

        # Update the online network via SGD.
        variables = self._online_network.trainable_variables
        tape = Horovod_Model.gradient_tape(tape)
        gradients = tape.gradient(loss, variables)
        self._optimizer.apply(gradients, variables)
        Horovod_Model.first(self._online_network, self._optimizer)
        
        # Update target network
        if update:
            for target, param in zip(self._target_network.trainable_variables, self._online_network.trainable_variables):
                target.assign(param)
        return td_error