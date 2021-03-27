from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, GaussianNoise, BatchNormalization, Flatten
from keras.optimizers import Adam


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
    layers.append(Dense(np.prod(self.env.action_space.nvec), activation='sigmoid')(layers[-1]))
    layers.append(Flatten()(layers[-1]))

    model = Model(inputs=layers[0], outputs=layers[-1])
    model.summary()
    print('', flush=True)

    # optimizer = self.candle.build_optimizer(self.optimizer, self.learning_rate, self.candle.keras_default_config())
    # model.compile(loss=self._huber_loss, optimizer=optimizer)
    return model
