import gym
import time
import numpy as np
import sys
import json
import exarl as erl
from exarl.base.comm_base import ExaComm



DEFAULT_X = np.pi
DEFAULT_Y = 1.0

class MyPendulum(gym.Env):

    def __init__(self):
        super().__init__()
        self.env_comm = ExaComm.env_comm
        self.env = gym.make("Pendulum-v1")
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        print('self.action_space', type(self.action_space), self.action_space)
        print('self.observation_space', type(self.observation_space), self.observation_space)
        
        
        
    def step(self, action):
        return self.env.step(action)

    def reset(self):
#        high = np.array([DEFAULT_X, DEFAULT_Y])
#        low = -high
#        self.env.state = self.env.np_random.uniform(low=low, high=high)
        ret = self.env.reset(seed=1)
        print("STATE",self.env.state)
        return ret

#    def render(self, mode='human', close=False):
#        return self.env.render()

    def set_env(self):
        print('Use this function to set hyper-parameters, if any')
