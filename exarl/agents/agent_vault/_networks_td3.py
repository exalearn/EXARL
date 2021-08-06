import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

class CriticModel(keras.Model):
    def __init__(self, fc_dims=[512, 512], name='Critic', activation_in='relu',activation_out=None):
        super(CriticModel,self).__init__()
        self.fc1_dims = fc_dims[0]
        self.fc2_dims = fc_dims[1]

        self.model_name = name
        self.activation_in = activation_in
        self.activation_out = activation_out
        self.fc1 = Dense(self.fc1_dims, activation=activation_in)
        self.fc2 = Dense(self.fc2_dims, activation=activation_in)
        self.q = Dense(1,activation=activation_out)

    def call(self, state, action):
        action_value = self.fc1(tf.concat([state, action], axis=1))
        action_value = self.fc2(action_value)

        return self.q(action_value)


class ActorModel(keras.Model):
    def __init__(self, fc_dims=[512,512], n_action=3, name='Actor', activation_in='relu',activation_out='tanh'):
        super(ActorModel,self).__init__()
        self.fc1_dims = fc_dims[0]
        self.fc2_dims = fc_dims[1]
        self.n_action = n_action
        self.model_name = name
        self.activation_in = activation_in
        self.activation_out = activation_out
        self.fc1 = Dense(self.fc1_dims, activation=activation_in)
        self.fc2 = Dense(self.fc2_dims, activation=activation_in)
        self.mu = Dense(self.n_action, activation=activation_out)

    def call(self, state):
        probability = self.fc1(state)
        probability = self.fc2(probability)

        mu = self.mu(probability)

        return mu
