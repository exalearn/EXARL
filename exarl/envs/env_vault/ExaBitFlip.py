import gym
from gym import spaces
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

    metadata = {'render.modes': ['human']}

    def __init__(self, bit_length=16, max_steps=100, mean_zero=False):
        super(BitFlip, self).__init__()
        self.env_comm = mpi_settings.env_comm
        if bit_length < 1:
            raise ValueError('bit_length must be >= 1, found {}'.format(bit_length))
        self.bit_length = bit_length

        self.max_steps = max_steps
        self.action_space = spaces.Discrete(bit_length)
        self.observation_space = spaces.Dict({
            'state': spaces.Box(low=0, high=1, shape=(bit_length, )),
            'goal': spaces.Box(low=0, high=1, shape=(bit_length, ))
        })
        self.reset()

    def _terminate(self):
        return (self.state == self.goal).all() or self.steps >= self.max_steps

    def _reward(self):
        # For sparse reward
        return -1 if (self.state != self.goal).any() else 0

    def step(self, action):
        # Flips the bit
        self.state[action] = int(not self.state[action])
        self.steps += 1

        time.sleep(0)  # Delay in seconds

        rank = self.env_comm.rank
        if rank == 0:
            N = 100
        else:
            N = None

        N = self.env_comm.bcast(N, root=0)
        myPI = computePI(N, self.env_comm)  # Calls python function
        # myPI = cp.compute_pi(N, self.env_comm) # Calls C++ function
        PI = self.env_comm.reduce(myPI, op=MPI.SUM, root=0)

        if self.env_comm.rank == 0:
            print(PI)  # Print PI for verification

        return self._get_obs(), self._reward(), self._terminate(), {}

    def _get_obs(self):

        return {
            'state': self.state,
            'goal': self.goal,
        }

    def __str__(self):
        return "State: {}, Goal: {}, Bit length: {}, Max Steps {}, Action Space {}"\
            .format(self.state, self.goal, self.bit_length, self.max_steps, self.action_space)

    def render(self, mode='human', close=False):
        # return self.env.render()
        pass

    def generate_states(self):
        return np.random.choice(2, self.bit_length)

    def reset(self):
        self.steps = 0
        self.state = self.generate_states()
        self.goal = self.generate_states()
        while (self.goal == self.state).all():
            self.goal = self.generate_states()
        return self.state, self.goal
