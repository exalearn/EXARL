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
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense

#TODO : Have to improve the model later 
class CriticModel(keras.Model):
    def __init__(self, n_actions, fc_dims=[512, 512], name='Critic', chkpt_dir='tmp/sac', activation_in='relu',activation_out=None):
        super(CriticModel,self).__init__()
        self.fc1_dims = fc_dims[0]
        self.fc2_dims = fc_dims[1]
        self.n_actions = n_actions

        self.model_name = name
        self.chkpt_dir = chkpt_dir
        self.activation_in = activation_in
        self.activation_out = activation_out
        self.fc1 = Dense(self.fc1_dims, activation=activation_in)
        self.fc2 = Dense(self.fc2_dims, activation=activation_in)
        self.q = Dense(1,activation=None)

    def __call__(self, state, action):
        action_value = self.fc1(tf.concat([state, action], axis=1))
        action_value = self.fc2(action_value)

        return self.q(action_value)
    
    # TODO: Remove repeatability
    def get_checkpoint_name(self, number=None):
        if number is None:
            file_name = self.model_name+'_sac'
            return os.path.join(self.chkpt_dir, file_name)
        file_name =  self.model_name+'_'+str(number)+'_sac'
        return os.path.join(self.chkpt_dir, file_name)


class ValueModel(keras.Model):
    def __init__(self, fc_dims=[256, 256], name='Value', chkpt_dir='tmp/sac', activation_in='relu',activation_out=None):
        super(ValueModel,self).__init__()
        self.fc1_dims = fc_dims[0]
        self.fc2_dims = fc_dims[1]

        self.model_name = name
        self.chkpt_dir = chkpt_dir
        self.activation_in = activation_in
        self.activation_out = activation_out
        self.fc1 = Dense(self.fc1_dims, activation=activation_in)
        self.fc2 = Dense(self.fc2_dims, activation=activation_in)
        self.v = Dense(1,activation=None)

    def __call__(self, state):
        state_value = self.fc1(state)
        state_value = self.fc2(state_value)

        return self.v(state_value)
    # TODO: Remove repeatability
    def get_checkpoint_name(self, number=None):
        if number is None:
            file_name = self.model_name+'_sac'
            return os.path.join(self.chkpt_dir, file_name)
        file_name =  self.model_name+'_'+str(number)+'_sac'
        return os.path.join(self.chkpt_dir, file_name)


class ActorModel(keras.Model):
    def __init__(self, max_action, fc_dims=[256,256], n_action=3, name='Actor', chkpt_dir='tmp/sac', activation_in='relu',activation_out=None, noise=1e-6):
        super(ActorModel,self).__init__()
        self.fc1_dims = fc_dims[0]
        self.fc2_dims = fc_dims[1]
        self.n_action = n_action
        self.model_name = name
        self.chkpt_dir = chkpt_dir
        self.max_action = max_action
        self.activation_in = activation_in
        self.activation_out = activation_out
        self.noise = noise #TODO: Change this to use the OAUNoise, might add this later
        self.fc1 = Dense(self.fc1_dims, activation=activation_in)
        self.fc2 = Dense(self.fc2_dims, activation=activation_in)
        self.mu = Dense(self.n_action, activation=activation_out)
        self.sigma = Dense(self.n_action, activation=activation_out)
    
    def __call__(self, state):
        probability = self.fc1(state)
        probability = self.fc2(probability)

        mu = self.mu(probability)
        sigma = self.sigma(probability)

        return mu, sigma
    # TODO: Remove repeatability
    def get_checkpoint_name(self, number=None):
        if number is None:
            file_name = self.model_name+'_sac'
            return os.path.join(self.chkpt_dir, file_name)
        file_name =  self.model_name+'_'+str(number)+'_sac'
        return os.path.join(self.chkpt_dir, file_name)


    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.__call__(state)
        probabilities = tfp.distributions.Normal(mu, sigma)
        if reparameterize:
            actions = probabilities.sample() #TODO: double check type of reparameterization
        else:
            actions = probabilities.sample()
        action = tf.math.tanh(actions)*self.max_action
        log_probs = probabilities.log_prob(actions)
        log_probs -= tf.math.log(1-tf.math.pow(action, 2) + self.noise) #noise to avoid taking log of zero
        log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)

        return action, log_probs





