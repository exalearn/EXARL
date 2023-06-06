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
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as tf_layer
from tensorflow.keras.regularizers import l1_l2
from gym.spaces.utils import flatdim

from exarl.agents.models.tf_model import Tensorflow_Model
from exarl.utils.globals import ExaGlobals

class LSTM(Tensorflow_Model):
    def __init__(self, observation_space, action_space, use_gpu=True):
        super(LSTM, self).__init__(observation_space, action_space, use_gpu)
        self.batch_size = ExaGlobals.lookup_params('batch_size')
        self.trajectory_length = ExaGlobals.lookup_params('trajectory_length')
        self.activation = ExaGlobals.lookup_params('activation')
        self.out_activation = ExaGlobals.lookup_params('out_activation')
        self.lstm_layers = ExaGlobals.lookup_params('lstm_layers')
        self.gauss_noise = ExaGlobals.lookup_params('gauss_noise')
        self.regularizer = ExaGlobals.lookup_params('regularizer')
        self.clipnorm = ExaGlobals.lookup_params('clipnorm')
        self.clipvalue = ExaGlobals.lookup_params('clipvalue')

        # self.optimizer = ExaGlobals.lookup_params('optimizer')
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss = ExaGlobals.lookup_params('loss')

    def _build(self):
        num_layers = len(self.lstm_layers)
        
        self._model = Sequential()
        self._model.add(tf_layer.LSTM(self.lstm_layers[0], activation=self.activation, 
                                      return_sequences=True, 
                                      input_shape=(self.trajectory_length, flatdim(self.observation_space))))
        self._model.add(tf_layer.BatchNormalization())
        self._model.add(tf_layer.Dropout(self.gauss_noise[0]))

        # loop over inner layers only
        for l in range(1, num_layers - 1):
            self._model.add(tf_layer.LSTM(self.lstm_layers[l], 
                                          activation=self.activation,
                                          return_sequences=True))
            self._model.add(tf_layer.Dropout(self.gauss_noise[l]))

        # special case for output layer
        l = num_layers = 1
        self._model.add(tf_layer.LSTM(self.lstm_layers[l], 
                                      activation=self.activation,
                                      kernel_regularizer=l1_l2(self.regularizer[0], 
                                      self.regularizer[1]),))
        self._model.add(tf_layer.Dropout(self.gauss_noise[l]))
        self._model.add(tf_layer.Dense(flatdim(self.action_space), activation=self.out_activation))
