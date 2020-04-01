import gym
import time
from mpi4py import MPI
import numpy as np
import sys
import json
import exarl as erl
from envs.env_vault.computePI import computePI as cp

def computePI(N,new_comm):
    h = 1.0 / N
    s = 0.0
    rank = new_comm.rank
    size = new_comm.size
    for i in range(rank, N, size):
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
        #print('Max steps: %s' % str(self._max_episode_steps))
        #self.env = gym.make('FrozenLake-v0')
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        time.sleep(0) # Delay in seconds
        ##
        
        # Compute PI with communicator splitting
        comm = MPI.COMM_WORLD
        world_rank = comm.Get_rank()
        
        if world_rank == 0:
            N = 100
        else:
            N = None
            
        N = comm.bcast(N, root=0)	
        color = int(world_rank/self.mpi_children_per_parent)
        newcomm = comm.Split(color, world_rank)
        #myPI = computePI(N, newcomm)
        myPI = cp.compute_pi(N, newcomm)
        PI = newcomm.reduce(myPI, op=MPI.SUM, root=0)
        newrank = newcomm.rank
        #if newrank == 0:
        #    print(PI)

        newcomm.Free()
        
        return next_state, reward, done, info

    def reset(self):
        self.env._max_episode_steps=self._max_episode_steps
        #print('Max steps: %s' % str(self._max_episode_steps))
        return self.env.reset()

    def render(self, mode='human', close=False):
        return self.env.render()
