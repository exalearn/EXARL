from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Input,GaussianNoise,BatchNormalization,LSTM
from keras.optimizers import Adam

def build_model(self):
    model = Sequential()
    model.add(LSTM(56, return_sequences=True,input_shape=(1, self.env.observation_space.shape[0])))
    model.add(GaussianNoise(0.1))
    #model.add(Dropout(0.2))                                                                                                  
    model.add(LSTM(56, return_sequences=True))
    model.add(GaussianNoise(0.1))
    #model.add(Dropout(0.2))                                                                                                  
    model.add(LSTM(56))
    model.add(GaussianNoise(0.1))
    model.add(Dense(self.env.action_space.n,))
    opt = Adam(lr=1e-4)
    model.compile(loss='mean_squared_error', optimizer=opt)
    model.summary()
    return model
