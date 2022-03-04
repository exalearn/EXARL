import time
import sys
import gym
import json
import exarl as erl
import numpy as np
from gym import spaces
import exarl.utils.candleDriver as cd

class GymSpaceTest(gym.Env):

    def __init__(self):
        super().__init__()

        high = np.array([1, 1, 1], dtype=np.float64)

        spaceDict = {
            "Discrete": spaces.Discrete(5),
            "Box_One": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float64),
            "Box_Two": spaces.Box(low=-1, high=1, shape=(1, 2), dtype=np.float64),
            "Box": spaces.Box(low=-high, high=high, dtype=np.float64),
            "MultiBinary": spaces.MultiBinary([2, 3]),
            "MultiDiscrete": spaces.MultiDiscrete([3, 2]),
            "Dict": spaces.Dict({
                "discrete": spaces.Discrete(100),
                "box": spaces.Box(low=-1, high=1, shape=(3, 4), dtype=np.float64),
                "multiBinary": spaces.MultiBinary([2, 3]),
                "multiDiscrete": spaces.MultiDiscrete([3, 2])
            })
        }

        boolDict = {
            "True": True,
            "true": True,
            "False": False,
            "false": False
        }

        actSpace = spaceDict[cd.run_params['action_space']]
        obvSpace = spaceDict[cd.run_params['observation_space']]
        actTuple = boolDict[cd.run_params['action_Tuple']]
        obvTuple = boolDict[cd.run_params['observation_Tuple']]

        if actTuple:
            self.action_space = spaces.Tuple((actSpace, actSpace))
        else:
            self.action_space = actSpace

        if obvTuple:
            self.observation_space = spaces.Tuple((obvSpace, obvSpace))
        else:
            self.observation_space = obvSpace

        self.initial_state = self.observation_space.sample()
        self.score = 0

        print("ACTION SPACE", type(self.action_space), type(self.action_space.sample()))

    def step(self, action):
        next_state = self.observation_space.sample()
        # while True:
        #     try:
        #         gym.spaces.utils.flatten(self.observation_space, next_state)
        #         print("GOOD")
        #         break
        #     except:
        #         print("BAD", next_state)
        #         next_state = self.observation_space.sample()

        print("ACTION", type(action), action)
        print("OBSERVATION", type(self.observation_space), next_state)

        self.score += 1
        reward = self.score
        done = False
        return next_state, reward, done, {}

    def reset(self):
        return self.initial_state
