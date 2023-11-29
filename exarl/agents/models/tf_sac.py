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
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda
from gym.spaces.utils import flatdim
from exarl.agents.models.tf_model import Tensorflow_Model
from exarl.utils.globals import ExaGlobals

class SoftActor(Tensorflow_Model):
    def __init__(self, observation_space, action_space, use_gpu=True):
        super(Actor, self).__init__(observation_space, action_space, use_gpu)
        self.batch_size = ExaGlobals.lookup_params('batch_size')
        self.actor_dense = ExaGlobals.lookup_params('actor_dense')
        self.actor_dense_act = ExaGlobals.lookup_params('actor_dense_act')
        self.actor_out_act = ExaGlobals.lookup_params('actor_out_act')

        assert len(self.actor_dense) >= 1, "Must have at least one actor_dense layer: {}".format(len(self.actor_dense))

        self.loss = None
        self.actor_lr = ExaGlobals.lookup_params('actor_lr')
        self.optimizer = tf.keras.optimizers.Adam(self.actor_lr)
        
        self.upper_bound = action_space.high
        self.lower_bound = action_space.low
        self.n_actions   = action_space.shape[0]

    def _build(self):
        last_init = tf.random_uniform_initializer()
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        layers = []
        layers.append(Input(shape=(flatdim(self.observation_space),), batch_size=self.batch_size))
        for i in range(len(self.actor_dense)):
            layers.append(Dense(self.actor_dense[i], activation=self.actor_dense_act)(layers[-1]))
        layers.append(Dense(self.n_actions, activation=self.actor_out_act, kernel_initializer=last_init)(layers[-1]))
        layers.append(Lambda(lambda i: i * (self.upper_bound - self.lower_bound) + self.lower_bound)(layers[-1]))

        layers.append(Dense(self.n_actions, kernel_initializer=last_init)(layers[-3]))
        layers.append(Lambda(lambda x: tf.exp(x) )(layers[-1]))
        self._model = Model(inputs=layers[0], outputs=[layers[-3], layers[-1]])

    def _compile(self):
        """
        This internal function compiles a tf model.
        """
        with tf.device(self._device):
            self._build()
            self._model.compile(loss=self.loss, optimizer=self.optimizer, jit_compile=self.enable_xla)

class SoftCritic(Tensorflow_Model):
    def __init__(self, observation_space, action_space, use_gpu=True):
        super(Critic, self).__init__(observation_space, action_space, use_gpu)
        self.batch_size = ExaGlobals.lookup_params('batch_size')

        self.critic_state_dense = ExaGlobals.lookup_params('critic_state_dense')
        self.critic_state_dense_act = ExaGlobals.lookup_params('critic_state_dense_act')
        self.critic_action_dense = ExaGlobals.lookup_params('critic_action_dense')
        self.critic_action_dense_act = ExaGlobals.lookup_params('critic_action_dense_act')

        self.critic_concat_dense = ExaGlobals.lookup_params('critic_concat_dense')
        self.critic_concat_dense_act = ExaGlobals.lookup_params('critic_concat_dense_act')

        assert len(self.critic_state_dense) >= 1, "Must have at least one critic_state_dense layer: {}".format(len(self.critic_state_dense))
        assert len(self.critic_action_dense) >= 1, "Must have at least one critic_action_dense layer: {}".format(len(self.critic_action_dense))
        assert len(self.critic_concat_dense) >= 1, "Must have at least one critic_concat_dense layer: {}".format(len(self.critic_concat_dense))

        self.loss = None
        self.critic_lr = ExaGlobals.lookup_params('critic_lr')
        self.optimizer = tf.keras.optimizers.Adam(self.critic_lr)

    def _build(self):
        state_layers = []
        state_layers.append(Input(shape=(flatdim(self.observation_space),), batch_size=self.batch_size))
        for i in range(len(self.critic_state_dense)):
            state_layers.append(Dense(self.critic_state_dense[i], activation=self.critic_state_dense_act)(state_layers[-1]))

        action_layers = []
        action_layers.append(Input(shape=(flatdim(self.action_space),), batch_size=self.batch_size))
        for i in range(len(self.critic_action_dense)):
            action_layers.append(Dense(self.critic_action_dense[i], activation=self.critic_action_dense_act)(action_layers[-1]))

        action_sd_layers = []
        action_sd_layers.append(Input(shape=(flatdim(self.action_space),), batch_size=self.batch_size))
        for i in range(len(self.critic_action_dense)):
            action_sd_layers.append(Dense(self.critic_action_dense[i], activation=self.critic_action_dense_act)(action_sd_layers[-1]))

        concat_layers = []
        concat_layers.append(Concatenate()([state_layers[-1], action_layers[-1], action_sd_layers[-1]]))
        for i in range(len(self.critic_concat_dense)):
            concat_layers.append(Dense(self.critic_concat_dense[i], activation=self.critic_concat_dense_act)(concat_layers[-1]))
        
        concat_layers.append(Dense(1)(concat_layers[-1]))
        self._model = Model([state_layers[0], action_layers[0], action_sd_layers[0]], concat_layers[-1])
