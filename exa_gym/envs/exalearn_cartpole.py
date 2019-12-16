import gym
from gym import error, spaces, utils
from gym.utils import seeding
import time
from mpi4py import MPI
import numpy as np
import sys

class ExaCartpole(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.env = gym.make('CartPole-v0')
        #self.env = gym.make('FrozenLake-v0')
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        time.sleep(0) # Delay in seconds
        # Compute PI with dynamic process spawning
        worker = './exa_gym/envs/cpi.py'
        comm = MPI.COMM_SELF.Spawn(sys.executable,
                                   args=[worker],
                                   maxprocs=2)

        N = np.array(100, 'i')
        comm.Bcast([N, MPI.INT], root=MPI.ROOT)
        PI = np.array(0.0, 'd')
        comm.Reduce(None, [PI, MPI.DOUBLE],op=MPI.SUM, root=MPI.ROOT)
        print(PI)

        comm.Disconnect()

        return next_state, reward, done, info

    def reset(self):
        return self.env.reset()

    def render(self, mode='human', close=False):
        return self.env.render()
