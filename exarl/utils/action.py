import sys
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

'''
This Action class convert a continuous valued array to its discretized array
using discrete uniform distribution

# How to use this class
  # Set the class with the following three required arrays
    action = Action(
        arrLower=arrLower,            # an arry containing lower value for each dimension
        arrUpper=arrUpper,            # an arry containing upper value for each dimension
        arrNumClasses=arrNumClasses)  # an arry containing # of disretized classes in each dimension
  # Convert the continuous valued array (arrContAction) to its discretized array (arrDiscAction)
  # using discrete uniform distribution
    arrDiscAction = action.descretize(arrContAction)
'''

class Action(object):

    '''
    A user set the following inputs:
    * arrLower: an arry containing lowver value for each dimension
    * arrUpper: an arry containing upper value for each dimension
    * arrNumClasses: an arry containing # of disretized classes in each dimension
    '''
    def __init__(self, arrLower, arrUpper, arrNumClasses):

        self.arrLower = arrLower  # a list containing lower value for each dimension
        self.arrUpper = arrUpper  # a list containing upper value for each dimension

        self.arrNumClasses = arrNumClasses  # # of disretized classes in each dimension

        self.dim_action = 0  # dimension of the actions (arrUpper-arrUpper)

        self.arrIntervals = []  # action interaval array

        self.arrDiscAction = []  # discretized action array

        self.setDimAction()

        self.checkLowerUpperArray()

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

        self.dim_action = len(self.arrNumClasses)  # set the dimension of the action array

        logging.debug(f"dim_action: {self.dim_action}")

    # validate the lower and upper arrays
    def checkLowerUpperArray(self):

        for i in range(self.dim_action):
            if (self.arrLower[i]) > (self.arrUpper[i]):
                print("ERROR: arrLower[i] <= arrUpper[i] for all i!")
                sys.exit()

    # set interval for each action dimension, arrIntervals
    def setArrIntervals(self):

        self.arrIntervals = np.zeros(self.dim_action)

        for i in range(self.dim_action):
            self.arrIntervals[i] = (
                self.arrUpper[i] - self.arrLower[i]) / self.arrNumClasses[i]

        logging.debug(f"arrIntervals: {self.arrIntervals}", )

    # discreteize continuous action using discrete uniform distribution
    # Input:  arrContAction: 1D arry of continuous action values
    # Output: arrDiscValues: 1D arry of discritizecd action values
    def descretize(self, arrContAction):

        if (len(arrContAction) != self.dim_action):
            print("ERROR: The dimension of actions, dim_action, "
                  "and cont_action should be equal!")
            sys.exit()

        logging.debug(f"continuous action array: {arrContAction}")

        self.arrDiscAction = np.zeros(self.dim_action)

        for i in range(self.dim_action):

            self.arrDiscAction[i] = (
                arrContAction[i] - self.arrLower[i]) // self.arrIntervals[i]

            if self.arrDiscAction[i] == self.arrNumClasses[i]:
                self.arrDiscAction[i] -= 1

        logging.debug(f"discretized action array: {self.arrDiscAction}")

        return self.arrDiscAction
