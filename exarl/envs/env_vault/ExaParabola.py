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
import gym
import time
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import json
import exarl as erl
# from envs.env_vault.computePI import computePI as cp
import exarl.mpi_settings as mpi_settings

class ExaParabola(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        self.env_comm = mpi_settings.env_comm
        
        # Set up the parabola
        self.a = 2
        self.b = 4
        self.c = 1
        self.f = lambda x: self.a*x**2.0 + self.b*x + self.c
        self.time_steps = 100

        self.x_min = -self.b/(2.0*self.a)
        self.y_min = self.f(self.x_min)

        self.high = 2
        self.low = -self.high

        # For use in render function
        self.x = np.linspace(self.low, self.high, 100)
        self.y = self.f(self.x)

        # Define action and observation space
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=np.array([self.low]), high=np.array([self.high]), dtype=np.float64)
        self.state = [random.uniform(self.low, self.high)]

    def step(self, action):
        
        action_step_size = 0.1

        if action == 1:
            self.state[0] += action_step_size
        else:
            self.state[0] -= action_step_size

        self.time_steps -= 1

        y_pred = self.f(self.state[0])
        y_diff = abs(self.y_min - y_pred)

        tol = 0.1
        reward = 1.0 - y_diff**2.0

        if y_diff <= tol:
            done = True
            return self.state, reward, done, {}

        if self.time_steps <= 0 or y_diff <= tol:
            done = True
        else:
            done = False

        return self.state, reward, done, {}

    def reset(self):
        # self.env._max_episode_steps=self._max_episode_steps
        # print('Max steps: %s' % str(self._max_episode_steps))
        self.state = [random.uniform(self.low, self.high)]
        self.time_steps = 100
        return self.state

    def render(self, mode='human', close=False):
#        plt.clf()
#        plt.plot(self.x, self.y)
#        plt.plot(self.x_min, self.y_min, 'g*', label='minimum value')
#        plt.plot(self.state, self.f(self.state), 'r*', label='current state')
#        plt.legend()
#        plt.show()
        return

    def set_env(self):
        print('Use this function to set hyper-parameters, if any')
