import gym
import time
from mpi4py import MPI
import numpy as np
import sys
import json
import exarl as erl
import random
# from envs.env_vault.computePI import computePI as cp
import exarl.mpi_settings as mpi_settings


def computePI(N, new_comm):
    h = 1.0 / N
    s = 0.0
    rank = new_comm.rank
    size = new_comm.size
    for i in range(rank, N, size):
        x = h * (i + 0.5)
        s += 4.0 / (1.0 + x**2)
    return s * h


class BitFlip(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 30}

    def __init__(self, bit_length=16, max_steps=100, mean_zero=False):
        super(BitFlip, self).__init__()
        self.env_comm = mpi_settings.env_comm
        if bit_length < 1:
            raise ValueError('bit_length must be >= 1, found {}'.format(bit_length))
        self.bit_length = bit_length
        self.mean_zero = mean_zero

        self.max_steps = max_steps
        self.action_space = spaces.Discrete(bit_length)
        self.observation_space spaces.Dict({
            'state': spaces.Box(low=0, high=1, shape=(bit_length, )),
            'goal': spaces.Box(low=0, high=1, shape=(bit_length, ))
        })
        self.reset()

    def _terminate(self):
        return (self.state == self.goal).all() or self.steps >= self.max_steps

    def _reward(self):
        return -1 if (self.state != self.goal).any() else 0

    def step(self, action):
        self.state[action] = int(not self.state[action])
        self.steps += 1

        return self._get_obs(), self._reward(), self._terminate(), {}

    def _mean_zero(self, x):
        if self.mean_zero:
            return (x - 0.5) / 0.5
        return x

    def _get_obs(self):

        return {
            'state': self._mean_zero(self.state),
            'goal' : self._mean_zero(self.goal),
        }
    
    def _render(self, mode='human', close=False):
        pass
    
