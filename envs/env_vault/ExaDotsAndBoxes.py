import time
import sys
import gym
import exarl as erl
import dotsandboxes as dab
import numpy as np
from gym import spaces

class ExaDotsAndBoxes(gym.Env):

    def __init__(self):
        super().__init__()
        self.action_space = spaces.MultiDiscrete(12)
        self.observation_space = spaces.MultiDiscrete(20)
        self.dumbScore=0

        # self.action_space = spaces.Tuple( ( spaces.MultiBinary([2,3]), spaces.MultiBinary([3,2]) ) )
        # self.observation_space = spaces.Tuple( ( spaces.MultiBinary([2,3]), spaces.MultiBinary([3,2]), spaces.MultiBinary([2,2]), spaces.MultiBinary([2,2]) ) )

    def step(self, action):
        print("ACTION", type(action), action)
        # reward = dab.step(0,3)
        # done = dab.done()
        # next_state = dab.state()

        next_state = self.observation_space.sample()
        self.dumbScore+=1
        reward = self.dumbScore
        done = False
        return next_state, reward, done, {}

    def reset(self):
        # dab.reset()
        # dab.state()
        return self.observation_space.sample()

    def set_env(self):
        print('Use this function to set hyperparameters, if any')
