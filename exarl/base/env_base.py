# © (or copyright) 2020. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
# National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
# Department of Energy/National Nuclear Security Administration. All rights in the program are
# reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
# Security Administration. The Government is granted for itself and others acting on its behalf a
# nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
# derivative works, distribute copies to the public, perform publicly and display publicly, and
# to permit others to do so.
import os
import numpy as np
from gym import spaces
from gym import Wrapper
from exarl.base.comm_base import ExaComm
from exarl.utils.globals import ExaGlobals

class ExaEnv(Wrapper):
    def __init__(self, env, **kwargs):
        super(ExaEnv, self).__init__(env)
        self.env = env
        self.env.workflow_episode = 0
        self.env.workflow_step = 0

        # Use relative path not absolute
        self.base_dir = os.path.dirname(__file__)
        self.env_comm = ExaComm.env_comm

        self.action_type = ExaGlobals.lookup_params('convert_action_type')
        self.original_env_type = type(self.env.action_space)

        if self.action_type == "Discrete":
            if isinstance(self.env.action_space, spaces.Box):
                self.old_action_space = self.env.action_space
                self.num_discrete_steps = int(ExaGlobals.lookup_params('num_discrete_step'))

                self.min = self.env.action_space.low
                self.increment = (self.env.action_space.high - self.env.action_space.low) / self.num_discrete_steps

                self.flat_dim = 1
                for x in self.env.action_space.shape:
                    self.flat_dim *= x

                if self.flat_dim == 1:
                    self.old_contains = self.env.action_space
                    self.env.action_space = spaces.Discrete(self.num_discrete_steps)
                else:
                    self.old_contains = self.env.action_space.contains
                    self.env.action_space = spaces.MultiDiscrete(self.flat_dim * [self.num_discrete_steps])
                print("Converted Action Space to Discrete", self.flat_dim)

            else:
                self.action_type = None

        elif self.action_type == "Continuous":
            if isinstance(self.env.action_space, spaces.Discrete):
                # JS: newer versions of gym have start
                try:
                    self.min = self.env.action_space.start * np.ones((1,), dtype=int)
                except AttributeError:
                    self.min = np.zeros((1,), dtype=int)

                self.old_action_space = self.env.action_space
                self.env.action_space = spaces.Box(low=0, high=self.env.action_space.n - 1, shape=(1,))
                self.unpack_action = True
                print("Converted Action Space to Continuous Box 1")

            elif isinstance(self.env.action_space, spaces.MultiDiscrete):
                # JS: newer versions of gym have start
                try:
                    self.min = self.env.action_space.start * np.ones(self.env.action_space.shape(), dtype=int)
                except AttributeError:
                    self.min = np.zeros(self.env.action_space.shape(), dtype=int)

                self.old_action_space = self.env.action_space
                self.env.action_space = spaces.Box(low=0, high=self.env.action_space.n - 1, shape=self.env.action_space.shape())
                self.unpack_action = False
                print("Converted Action Space to Continuous Box N")

            else:
                self.action_type = None

    def set_episode_count(self, episode_count):
        '''
        Method to keep track of episode count in the env
        '''
        self.env.workflow_episode = episode_count

    def set_step_count(self, step_count):
        '''
        Method to keep track of step per episode in the env
        '''
        self.env.workflow_step = step_count

    def set_results_dir(self, results_dir):
        '''
        Default method to save environment specific information
        '''
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        # Top level directory
        self.results_dir = results_dir

    def swap_action_spaces(self):
        # print(self.env.workflow_episode, self.env.workflow_step, "Old:", type(self.old_action_space), "New:", type(self.env.action_space), flush=True)
        temp = self.env.action_space
        self.env.action_space = self.old_action_space
        self.old_action_space = temp

    def step(self, action):
        if self.action_type == "Discrete":
            self.swap_action_spaces()
            ret = self.env.step(action * self.increment + self.min)
            self.swap_action_spaces()

        elif self.action_type == "Continuous":
            new_action = action.astype(int) + self.min
            if self.unpack_action:
                new_action = new_action.item(0)

            self.swap_action_spaces()
            ret = self.env.step(new_action)
            self.swap_action_spaces()

        else:
            ret = self.env.step(action)
        return ret

