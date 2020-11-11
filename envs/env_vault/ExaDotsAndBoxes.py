import time
import sys
import gym
import dotsandboxes as dab
import exarl as erl
import numpy as np
from gym import spaces

class ExaDotsAndBoxes(gym.Env):

    def __init__(self, **kwargs):
        super().__init__()
        self.dim = kwargs['boardSize']
        self.initMoves = kwargs['initialMoves']

        # 2 * Dimension * (Dimension - 1) + 2 (Dimension - 1)^2
        self.numLines = 2 * self.dim * (self.dim - 1)
        self.spaceLen = self.numLines + 2 * (self.dim - 1) * (self.dim - 1) + 2
        
        self.numBoxes = (self.dim - 1) * (self.dim - 1)
        self.minScore = -1 * self.numBoxes
        self.maxScore = self.numBoxes

        assert(self.dim > 1)
        assert(self.initMoves < self.numLines)
        print("Setting:", self.dim, self.initMoves)
        dab.setParams(self.dim, self.initMoves)

        self.observation_space = spaces.MultiBinary(self.spaceLen)
        self.action_space = spaces.Discrete(self.numLines)

    def step(self, action):
        
        assert(action < self.numLines)
        
        reward = dab.step(action)
        if reward < self.minScore:
            reward = self.minScore - 1
            done = True
        else:
            if not dab.done() and not dab.player1Turn():
                dab.oppenent(True)
            done = dab.done()
            if not done:
                assert(dab.player1Turn())
            reward = dab.score()[0]   

        next_state = dab.state()
        return next_state, reward, done, {}

    def reset(self):
        dab.reset()
        return dab.state()