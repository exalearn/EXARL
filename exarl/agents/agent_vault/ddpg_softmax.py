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
import numpy as np
import tensorflow as tf
from copy import deepcopy

import exarl
from exarl.utils.globals import ExaGlobals
from exarl.agents.replay_buffers.buffer import Buffer
from exarl.agents.replay_buffers.nStep_buffer import nStepBuffer
from exarl.utils.OUActionNoise import OUActionNoise
from exarl.agents.models.tf_model import Tensorflow_Model
from exarl.utils.introspect import introspectTrace

logger = ExaGlobals.setup_logger(__name__)

# https://github.com/keras-team/keras-io/blob/master/examples/rl/ddpg_pendulum.py

class DDPGSoftmax(exarl.ExaAgent):
    """
    Deep deterministic policy gradient agent.
    Inherits from ExaAgent base class.
    """

    def __init__(self, env, is_learner):
        """DDPG constructor

        Args:
            env (OpenAI Gym environment object): env object indicates the RL environment
            is_learner (bool): Used to indicate if the agent is a learner or an actor
        """
        self.is_learner = is_learner
        
        self.upper_bound = env.action_space.high
        self.lower_bound = env.action_space.low

        self.gamma = ExaGlobals.lookup_params('gamma')
        self.tau = ExaGlobals.lookup_params('tau')
        self.batch_size = ExaGlobals.lookup_params('batch_size')

        # Ornstein-Uhlenbeck process
        self.ou_noise = OUActionNoise(mean=np.zeros(env.action_space.shape), std_deviation=np.array([0.2]))

        # Experience data
        self.horizon    = ExaGlobals.lookup_params('horizon')
        assert self.horizon >= 1, "Invalid Horizon Value: " + str(self.horizon)
        if self.horizon == 1:
            self._replay = Buffer.create(observation_space=env.observation_space, action_space=env.action_space)
        else:
            self._replay = nStepBuffer(ExaGlobals.lookup_params('buffer_capacity'), self.horizon, self.gamma, observation_space=env.observation_space, action_space=env.action_space)

        self.actor = Tensorflow_Model.create("ActorSoftmax", 
                                              observation_space=env.observation_space, 
                                              action_space=env.action_space, 
                                              use_gpu=self.is_learner)
        self.critic = Tensorflow_Model.create("Critic", 
                                              observation_space=env.observation_space, 
                                              action_space=env.action_space, 
                                              use_gpu=self.is_learner)
        self.target_actor = Tensorflow_Model.create("ActorSoftmax", 
                                              observation_space=env.observation_space, 
                                              action_space=env.action_space, 
                                              use_gpu=self.is_learner)
        self.target_critic = Tensorflow_Model.create("Critic", 
                                              observation_space=env.observation_space, 
                                              action_space=env.action_space, 
                                              use_gpu=self.is_learner)
        
        self.actor.init_model()
        self.critic.init_model()
        self.actor.print()
        self.critic.print()
        self.target_actor.init_model()
        self.target_critic.init_model()

        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        self._forward = tf.function(self.actor)

    @introspectTrace()
    def action(self, state):
        """Returns sampled action with added noise

        Args:
            state (list or array): Current state of the system

        Returns:
            action (list or array): Action to take
            policy (int): random (0) or inference (1)
        """
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        prediction = self._forward(tf_state)
        sampled_actions = tf.squeeze(prediction)

        noise = self.ou_noise()
        sampled_actions_wn = sampled_actions.numpy() + noise
        legal_action = np.clip(sampled_actions_wn, self.lower_bound, self.upper_bound)
        legal_action = (legal_action + 1.e-10) / np.sum(legal_action + 1.e-10)
        return legal_action, 0

    def remember(self, state, action, reward, next_state, done):
        self._replay.store(state, action, reward, next_state, done)

    def has_data(self):
        """
        Indicates if the buffer has data

        Returns:
            bool: True if buffer has data
        """
        return self._replay.size > 0

    def _prep_data(self, data):
        data[2] = data[2].reshape((len(data[2]), 1))
        for i in range(4):
            data[i] = tf.convert_to_tensor(data[i])
        data[2] = tf.cast(data[2], dtype=tf.float32)
        return data

    @introspectTrace()
    def generate_data(self):
        ret = None
        if self.has_data():
            data = self._replay.sample(self.batch_size)
            ret = self._prep_data(data)
        yield ret

    @introspectTrace()
    def train(self, batch):
        """
        Train the NN

        Args:
            batch (list): sampled batch of experiences
        """
        if batch is not None:
            self.update_grad(batch[0], batch[1], batch[2], batch[3], batch[4])

    @tf.function
    def update_grad(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + (1.0 - tf.cast(done_batch[:,None], tf.float32))*self.gamma * self.target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.critic([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(
            zip(critic_grad, self.critic.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor(state_batch, training=True)
            critic_value = self.critic([state_batch, actions], training=True)
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables)
        )

        for (a, b) in zip(self.target_actor.variables, self.actor.variables):
            a.assign(b * self.tau + a * (1 - self.tau))

        for (a, b) in zip(self.target_critic.variables, self.critic.variables):
            a.assign(b * self.tau + a * (1 - self.tau))

    def get_weights(self):
        """Get weights from target model

        Returns:
            weights (list): target model weights
        """
        return self.actor.get_weights()

    def set_weights(self, weights):
        """Set model weights

        Args:
            weights (list): model weights
        """
        self.actor.set_weights(weights)

    def train_return(self, args):
        pass
