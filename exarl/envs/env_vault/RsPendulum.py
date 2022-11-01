from ast import excepthandler
from lib2to3.pytree import type_repr
import gym
import numpy as np
import exarl as erl
from exarl.base.comm_base import ExaComm

class RsPendulum(gym.Env):
    def __init__(self):
        super().__init__()
        self.env_comm = ExaComm.env_comm
        try:
            self.env = gym.make("Pendulum-v1")
        except:
            print("Pendulum-v1 version is not present setting Pendulum-v0")
            self.env = gym.make("Pendulum-v0")
            
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        self.last_u = None
        self.last_seed_episode = 0
        self.last_obs = self.env.reset()
        self.last_state = self.env.state

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        # JS: Maybe make sure this is right for when we are not getting
        # contiguous episodes...
        obs = self.env.reset()
        if self.workflow_episode % 2 == 0 and self.workflow_episode >= 2:
            self.last_seed_episode = self.workflow_episode
            self.last_obs = obs
            self.last_state = self.env.state
        else:
            self.env.state = self.last_state
        return self.last_obs
