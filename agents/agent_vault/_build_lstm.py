from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, GaussianNoise, BatchNormalization, LSTM
from keras.optimizers import Adam
from keras.regularizers import l1, l2, l1_l2
import numpy as np

def build_model(self):

    num_layers = len(self.lstm_layers)

    model = Sequential()
    # special case for input layer
    model.add(LSTM(self.lstm_layers[0], activation=self.activation,
                   return_sequences=True, input_shape=(1, self.env.observation_space.shape[0])))
    model.add(BatchNormalization())
    model.add(GaussianNoise(self.gauss_noise[0]))

    # loop over inner layers only
    for l in range(1, num_layers - 1):
        model.add(LSTM(self.lstm_layers[l], activation=self.activation,
                       return_sequences=True))
        model.add(GaussianNoise(self.gauss_noise[l]))

    # special case for output layer
    l = num_layers = 1
    model.add(LSTM(self.lstm_layers[l], activation=self.activation,
                   kernel_regularizer=l1_l2(self.regularizer[0], self.regularizer[1]),
                   ))
    model.add(GaussianNoise(self.gauss_noise[l]))
    model.add(Dense(np.prod(self.env.action_space.nvec), activation=self.out_activation))
    opt = Adam(lr=1e-3)  # ,clipnorm=1.0, clipvalue=0.5)
    # opt = self.candle.build_optimizer(self.optimizer, self.learning_rate,
    #                                  #clipnorm= self.clipnorm, clipvalue = self.clipvalue,
    #                                  self.candle.keras_default_config())

    # model.compile(loss=self.loss, optimizer=opt)
    # model.summary()
    return model
