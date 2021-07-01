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
import exarl.mpi_settings as mpi_settings
import exarl.utils.candleDriver as cd

class ExaParabola(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        self.env_comm = mpi_settings.env_comm

        # Load action step size, tolerance from config file
        self.action_step_size = cd.run_params['action_step_size']
        self.tol = cd.run_params['tolerance']

        # Set up the parabola
        self.a = cd.run_params['parabola_variable_a']
        self.b = cd.run_params['parabola_variable_b']
        self.c = cd.run_params['parabola_variable_c']
        self.f = lambda x: self.a*x**2.0 + self.b*x + self.c

        try:
            self.x_min = -self.b/(2.0*self.a)
        except:
            print('Parabola variable "a" must be non-zero.')
            sys.exit()

        self.y_min = self.f(self.x_min)

        # Define action and observation space
        self.high = cd.run_params['max_range']
        self.low = -self.high
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=np.array([self.low]), high=np.array([self.high]), dtype=np.float64)

        # Initialize starting state
        self.start_state_value = cd.run_params['start_state_value']
        self.state = [self.start_state_value]

        # For use in render function
        # self.x = np.linspace(self.low, self.high, 100)
        # self.y = self.f(self.x)

        # Define action and observation space
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=np.array([self.low]), high=np.array([self.high]), dtype=np.float64)
        self.state = [0]#[random.uniform(self.low, self.high)]


    def step(self, action):


        if action == 1:
            self.state[0] += self.action_step_size
        else:
            self.state[0] -= self.action_step_size

        y_pred = self.f(self.state[0])
        y_diff = abs(self.y_min - y_pred)

        reward = 1.0 - y_diff**2.0

        if y_diff <= self.tol:
            done = True
        else:
            done = False

        return self.state, reward, done, {}

    def reset(self):
        self.state = [self.start_state_value]
        return self.state

    def render(self, mode='human', close=False):
#        plt.clf()
#        plt.plot(self.x, self.y)
#        plt.plot(self.x_min, self.y_min, 'g*', label='minimum value')
#        plt.plot(self.state, self.f(self.state), 'r*', label='current state')
#        plt.legend()
#        plt.show()
        pass
