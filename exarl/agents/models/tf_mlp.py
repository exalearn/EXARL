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
from tensorflow.keras.layers import Dense, Input, Flatten
from gym.spaces.utils import flatdim
from exarl.agents.models.tf_model import Tensorflow_Model
from exarl.utils.globals import ExaGlobals

class MLP(Tensorflow_Model):
    def __init__(self, observation_space, action_space, use_gpu=True):
        super(MLP, self).__init__(observation_space, action_space, use_gpu)
        self.batch_size = ExaGlobals.lookup_params('batch_size')
        self.dense = ExaGlobals.lookup_params('dense')
        self.activation = ExaGlobals.lookup_params('activation')
        self.out_activation = ExaGlobals.lookup_params('out_activation')
        self.loss = ExaGlobals.lookup_params('loss')
        # self.optimizer = ExaGlobals.lookup_params('optimizer')
        self.optimizer = tf.keras.optimizers.Adam()

    def _build(self):
        layers = []
        # Input: state
        state_input = Input(shape=(flatdim(self.observation_space),), batch_size=self.batch_size)
        layers.append(state_input)
        for i in range(len(self.dense)):
            layer_width = self.dense[i]
            layers.append(Dense(layer_width, activation=self.activation)(layers[-1]))
        # Output layer
        layers.append(Dense(flatdim(self.action_space), dtype='float32', activation=self.out_activation)(layers[-1]))
        layers.append(Flatten()(layers[-1]))
        self._model = Model(inputs=layers[0], outputs=layers[-1])
