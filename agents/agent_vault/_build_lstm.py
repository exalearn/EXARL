from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Input,GaussianNoise,BatchNormalization,LSTM
from keras.optimizers import Adam
from keras.regularizers import l2

def build_model(self):
    model = Sequential()
    model.add(LSTM(56, activation='tanh',
                   kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01),
                   return_sequences=True,input_shape=(1, self.env.observation_space.shape[0])))
    model.add(GaussianNoise(0.1))
    #model.add(Dropout(0.2))                                                                                                  
    model.add(LSTM(56, activation='tanh',
                   kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01),
                   return_sequences=True))
    model.add(GaussianNoise(0.1))
    #model.add(Dropout(0.2))                                                                                                  
    model.add(LSTM(56, activation='tanh',
                   kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(GaussianNoise(0.1))
    model.add(Dense(self.env.action_space.n,activation='linear'))
    opt = Adam(lr=1e-3,clipnorm=1.0, clipvalue=1.0)
    model.compile(loss='mean_squared_error', optimizer=opt)
    model.summary()
    return model
