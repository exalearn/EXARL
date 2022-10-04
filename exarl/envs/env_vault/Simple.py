import gym
import time
import numpy as np
import sys
import json
import exarl as erl
from exarl.base.comm_base import ExaComm

class Simple(gym.Env):

    def __init__(self):
        super().__init__()
        self.env_comm = ExaComm.env_comm

        self.action_space = gym.spaces.box.Box(low = np.array([-1]),
                                               high = np.array([1]),
                                               dtype = np.float32)

        self.observation_space = gym.spaces.box.Box(low = np.array([-np.inf]),
                                                    high = np.array([np.inf]),
                                                    dtype = np.float32)

        print('self.action_space', type(self.action_space), self.action_space)

        print('self.observation_space', type(self.observation_space), self.observation_space)
        self.state = 0.1
        self.reward = 0
        
    def step(self, action):
        x = action[0]
        self.state = x
        next_state = np.array([self.state]) 
        reward = -(self.state - 2)**2
        self.reward += reward
        print('action', type(action), action)        
        print('next state', type(next_state), next_state)
        print('reward', type(reward), reward)

        return next_state, reward, False, None

    def reset(self):
        # self.env._max_episode_steps=self._max_episode_steps
        # print('Max steps: %s' % str(self._max_episode_steps))
        print("Total reward was {}".format(self.reward))
        self.state = 0.1
        self.reward = 0
        return np.array([self.state])
        #return self.env.reset()

#    def render(self, mode='human', close=False):
#        return self.env.render()

    def set_env(self):
        print('Use this function to set hyper-parameters, if any')
