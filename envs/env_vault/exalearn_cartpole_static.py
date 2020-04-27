import gym
import time
from mpi4py import MPI
import numpy as np
import sys
import json
import exarl as erl

def computePI(N,new_comm):
    h = 1.0 / N
    s = 0.0
    rank = new_comm.rank
    size = new_comm.size
    for i in range(rank, N, size):
        x = h * (i + 0.5)
        s += 4.0 / (1.0 + x**2)
    return s * h

def computePI(N,comm=MPI.COMM_WORLD):
    h = 1.0 / N
    s = 0.0
    for i in range(N):
        x = h * (i + 0.5)
        s += 4.0 / (1.0 + x**2)
    return s * h

class ExaCartpoleStatic(gym.Env, erl.ExaEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, cfg='envs/env_vault/env_cfg/env_setup.json'):
        super().__init__(env_cfg=cfg)
        self._max_episode_steps = 0
        self.env = gym.make('CartPole-v0')
        self.env._max_episode_steps=self._max_episode_steps
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    
    def step(self, action, comm=MPI.COMM_WORLD):
        next_state, reward, done, info = self.env.step(action)
        myPI = computePI(100, comm)
        comm.reduce(myPI, op=MPI.SUM, root=0)

        return next_state, reward, done, info

    def reset(self):
        self.env._max_episode_steps=self._max_episode_steps
        return self.env.reset()

    def render(self, mode='human', close=False):
        return self.env.render()
