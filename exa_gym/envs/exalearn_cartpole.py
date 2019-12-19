import gym
from gym import error, spaces, utils
from gym.utils import seeding
import time
from mpi4py import MPI
import numpy as np
import sys
import json

class ExaCartpole(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, cfg='exa_gym/envs/env_cfg/env_setup.json'):
        self._max_episode_steps = 0
        self.env = gym.make('CartPole-v0')
        self.env._max_episode_steps=self._max_episode_steps
        #print('Max steps: %s' % str(self._max_episode_steps))
        #self.env = gym.make('FrozenLake-v0')
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.cfg = cfg
        with open(self.cfg) as json_file:
            data = json.load(json_file)
        
        self.num_child_per_parent = int(data['child_spawn_per_parent']) if 'child_spawn_per_parent' in data.keys() else 1
        self.worker = (data['worker_app']).lower() if 'worker_app' in data.keys() else "/exa_gym/envs/cpi.py"

        ##

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        time.sleep(0) # Delay in seconds
        ##
        
        # Compute PI with dynamic process spawning
       
        #parent_comm = MPI.Comm.Get_parent()
        spawn_comm = MPI.COMM_SELF.Spawn(sys.executable,
                                   args=[self.worker],
                                   maxprocs=self.num_child_per_parent)#.Merge()

        N = np.array(100, 'i')
        spawn_comm.Bcast([N, MPI.INT], root=MPI.ROOT)
        PI = np.array(0.0, 'd')
        spawn_comm.Reduce(None, [PI, MPI.DOUBLE],op=MPI.SUM, root=MPI.ROOT)
        #print(PI)

        spawn_comm.Disconnect()
        #comm_world.Disconnect()
        #comm_slave.Free()
        
        return next_state, reward, done, info

    def reset(self):
        self.env._max_episode_steps=self._max_episode_steps
        #print('Max steps: %s' % str(self._max_episode_steps))
        return self.env.reset()

    def render(self, mode='human', close=False):
        return self.env.render()
