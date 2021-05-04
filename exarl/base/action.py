import sys
import numpy as np


class Action(object):

    # A user set the following inputs:
    # arrLower: a list of lower value for each dimension
    # arrUpper: a list of upper value for each dimension
    # arrNumClasses: # of disretized classes in each dimension

    def __init__(self, arrLower, arrUpper, arrNumClasses):

        self.arrLower = arrLower
        self.arrUpper = arrUpper

        self.arrNumClasses = arrNumClasses

        self.dim_action = 0

        self.arrIntervals = []

        self.arrDiscAction = []

        self.debug = 0               # TODO: one can pass an argument

        self.setDimAction()

        self.setArrIntervals()

    # set the dimension of action space
    def setDimAction(self):

        if (len(self.arrLower) != len(self.arrUpper)):
            print("ERROR: The dimension of the arrLower and arrUpper should be equal!")
            sys.exit()

        if (len(self.arrLower) != len(self.arrNumClasses)):
            print("ERROR: The dimension of the arrLower, arrUpper, "
                  "and arrNumClasses should be equal!")
            sys.exit()

        self.dim_action = len(self.arrNumClasses)

        if (self.debug >= 10):
            print("dim_action: ", self.dim_action)

    # set interval for each action dimension, arrIntervals
    def setArrIntervals(self):

        self.arrIntervals = np.zeros(self.dim_action)

        for i in range(self.dim_action):
            self.arrIntervals[i] = (
                self.arrUpper[i] - self.arrLower[i]) / self.arrNumClasses[i]

        if (self.debug >= 10):
            print("arrIntervals: ", self.arrIntervals)

    # discreteize continuous action
    # Input:  arrContAction: 1D arry of continuous action values
    # Output: arrDiscValues: 1D arry of discritizecd action values
    def descretize(self, arrContAction):

        if (len(arrContAction) != self.dim_action):
            print("ERROR: The dimension of actions, dim_action, "
                  "and cont_action should be equal!")
            sys.exit()

        if (self.debug >= 10):
            print("cont_action: ", arrContAction)

        self.arrDiscAction = np.zeros(self.dim_action)

        for i in range(self.dim_action):
            self.arrDiscAction[i] = (
                arrContAction[i] - self.arrLower[i]) // self.arrIntervals[i]

            if self.arrDiscAction[i] == self.arrNumClasses[i]:
                self.arrDiscAction[i] -= 1

        if (self.debug >= 10):
            print("arrDiscValues: ", self.arrDiscAction)

        return self.arrDiscAction
