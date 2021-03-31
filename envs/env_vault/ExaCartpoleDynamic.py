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
import sys
import json
import exarl as erl


class ExaCartpoleDynamic(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, cfg='envs/env_vault/env_cfg/env_setup.json'):
        super().__init__(env_cfg=cfg)
        self._max_episode_steps = 0
        self.env = gym.make('CartPole-v0')
        self.env._max_episode_steps = self._max_episode_steps
        # print('Max steps: %s' % str(self._max_episode_steps))
        # self.env = gym.make('FrozenLake-v0')
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        time.sleep(0)  # Delay in seconds
        # Compute PI with dynamic process spawning

        # parent_comm = MPI.Comm.Get_parent()
        spawn_comm = MPI.COMM_SELF.Spawn(sys.executable,
                                         args=[self.worker],
                                         maxprocs=self.process_per_env)  # .Merge()

        N = np.array(100, 'i')
        spawn_comm.Bcast([N, MPI.INT], root=MPI.ROOT)
        PI = np.array(0.0, 'd')
        spawn_comm.Reduce(None, [PI, MPI.DOUBLE], op=MPI.SUM, root=MPI.ROOT)
        # print(PI)

        spawn_comm.Disconnect()
        # comm_world.Disconnect()
        # comm_slave.Free()

        return next_state, reward, done, info

    def reset(self):
        self.env._max_episode_steps = self._max_episode_steps
        # print('Max steps: %s' % str(self._max_episode_steps))
        return self.env.reset()

    def render(self, mode='human', close=False):
        return self.env.render()

    def set_env(self):
        print('Use this function to set hyperparameters, if any')
