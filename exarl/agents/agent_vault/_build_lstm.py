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


def build_model(self):

    num_layers = len(self.lstm_layers)

    model = Sequential()
    # special case for input layer
    model.add(LSTM(self.lstm_layers[0], activation=self.activation,
                   return_sequences=True, input_shape=(1, self.env.observation_space.shape[0])))
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
    model.add(Dense(self.env.action_space.n, activation=self.out_activation))

    # model.summary()
    print('', flush=True)
    return model
