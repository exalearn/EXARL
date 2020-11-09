import time
import sys
import gym
import json
import exarl as erl
import numpy as np
from gym import spaces


class GymSpaceTest(gym.Env):

    def __init__(self, **kwargs):
        super().__init__()
        envConfig = kwargs['params']

        spaceDict = {
            "Discrete": spaces.Discrete(7),
            "Box": spaces.Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32),
            "MultiBinary": spaces.MultiBinary([2,3]),
            "MultiDiscrete": spaces.MultiDiscrete([3,2]),
            "Dict": spaces.Dict({"position": spaces.Discrete(2), "velocity": spaces.Discrete(3)})
        }

        if envConfig["actionTuple"] == True:
            self.action_space = spaces.Tuple((spaceDict[envConfig["actionSpace"]], spaceDict[envConfig["actionSpace"]]))
        else:
            self.action_space = spaceDict[envConfig["actionSpace"]]
            
        if envConfig["observationTuple"] == True:
            self.observation_space = spaces.Tuple((spaceDict[envConfig["observationSpace"]], spaceDict[envConfig["observationSpace"]]))
        else:
            self.observation_space = spaceDict[envConfig["observationSpace"]]

        print(type(self.action_space), type(self.observation_space))

        self.initial_state = self.observation_space
        self.score = 0
        
    def step(self, action):

        print("ACTION", type(action), action)
        # assert( type(action) == type(self.action_space) )

        next_state = self.observation_space.sample()
        # print(type(next_state), type(self.observation_space))
        # assert( type(next_state) == type(self.observation_space) )

        self.score+=1
        reward = self.score
        done = False
        return next_state, reward, done, {}

    def reset(self):
        return self.initial_state
