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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM
from tensorflow.keras.regularizers import l1_l2
from gym.spaces.utils import flatdim

def build_model(self):

    num_layers = len(self.lstm_layers)

    model = Sequential()
    """ TODO: This input layer is not taking advantage of memory
    The input shape should be of the form (batch_size, sequence_size, feature_size)
    See the following for clarification:
    https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
    https://github.com/MohammadFneish7/Keras_LSTM_Diagram

    Batch size - how many sequences we are passing in
    Sequence size - sequence of events.  This is related to our memory
    Feature size - this is the flatten dimension of the observation space

    It seems we need to omit the batch size:
    https://stackoverflow.com/questions/44583254/valueerror-input-0-is-incompatible-with-layer-lstm-13-expected-ndim-3-found-n
    https://github.com/keras-team/keras/issues/7403

    Ultimately what needs to change is the 1 for sequence size.  The question is, how does this change the way we pass
    data between the learner and actors.  It would seem that instead of randomly picking out experiences, we would need
    to pick out a series of contiguous experiences.  In this case changing between lstm and mlp is not trivial.
    """
    model.add(LSTM(self.lstm_layers[0], activation=self.activation, return_sequences=True, input_shape=(1, flatdim(self.env.observation_space))))
    model.add(BatchNormalization())
    model.add(Dropout(self.gauss_noise[0]))

    # loop over inner layers only
    for l in range(1, num_layers - 1):
        model.add(LSTM(self.lstm_layers[l], activation=self.activation,
                       return_sequences=True))
        model.add(Dropout(self.gauss_noise[l]))

    # special case for output layer
    l = num_layers = 1
    model.add(LSTM(self.lstm_layers[l], activation=self.activation,
                   kernel_regularizer=l1_l2(self.regularizer[0], self.regularizer[1]),
                   ))
    model.add(Dropout(self.gauss_noise[l]))
    model.add(Dense(flatdim(self.env.action_space), activation=self.out_activation))

    # model.summary()
    # print('', flush=True)
    return model
