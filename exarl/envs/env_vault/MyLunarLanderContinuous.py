import gym
import numpy as np
import exarl as erl
from exarl.base.comm_base import ExaComm


class MyLunarLanderContinuous(gym.Env):
    def __init__(self):
        super().__init__()
        self.env_comm = ExaComm.env_comm
        
        self.env = gym.make("LunarLander-v2",continuous=True)
            
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        self.last_u = None
        self.last_seed_episode = 0
        self.last_obs = self.env.reset()
        # self.last_state = self.env.state

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()