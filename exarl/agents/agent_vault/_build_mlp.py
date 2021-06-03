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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Flatten


def build_model(self):
    # Input: state
    layers = []
    state_input = Input(shape=(1, self.env.observation_space.shape[0]))
    layers.append(state_input)
    length = len(self.dense)
    # for i, layer_width in enumerate(self.dense):
    for i in range(length):
        layer_width = self.dense[i]
        layers.append(Dense(layer_width, activation=self.activation)(layers[-1]))
    # output layer
    layers.append(Dense(self.env.action_space.n, activation=self.out_activation)(layers[-1]))
    layers.append(Flatten()(layers[-1]))

    model = Model(inputs=layers[0], outputs=layers[-1])
    # model.summary()
    print('', flush=True)

    return model
