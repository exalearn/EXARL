import time
import sys
import gym
import exarl as erl
import dotsandboxes as dab
import numpy as np
from gym import spaces

class ExaDotsAndBoxes(gym.Env):

    def __init__(self, **kwargs):
        super().__init__()
        self.dim = kwargs['boardSize']
        self.initMoves = kwargs['initialMoves']

        # 2 * Dimension * (Dimension - 1) + 2 (Dimension - 1)^2
        self.numLines = 2 * self.dim * (self.dim - 1)
        self.spaceLen = self.numLines + 2 * (self.dim - 1) * (self.dim - 1)

        assert(self.dim > 1)
        assert(self.initMoves < self.numLines)
        print("Setting:", self.dim, self.initMoves)
        dab.setParams(self.dim, self.initMoves)

        self.observation_space = spaces.MultiBinary(self.spaceLen)
        self.action_space = spaces.Discrete(self.numLines)

    def step(self, action):
        # print("ACTION", type(action), action)
        assert(action < self.numLines)

        rndAction = action
        if rndAction < self.numLines / 2:
            row = rndAction / (self.dim - 1)
            col = rndAction % (self.dim - 1)
            src = row * self.dim + col
            dst = src + 1
        else:
            rndAction -= self.numLines/2
            row = action / self.dim
            col = action % self.dim
            src = row * self.dim + col
            dst = src + self.dim

        print("Action:", action, "Src:", src, "Dst:", dst)
        reward = dab.step(int(src),int(dst))
        done = dab.done()
        next_state = dab.state()

        return next_state, reward, done, {}

    def reset(self):
        dab.reset()
        return dab.state()