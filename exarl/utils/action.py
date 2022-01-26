# -*- coding: utf-8 -*-
"""Convert a continuous valued array to its discretized array

This Action class convert a continuous valued array to its discretized array
using discrete uniform distribution

Example
-------
    action = Action(
        arrLower=arrLower,            
        arrUpper=arrUpper,            
        arrNumClasses=arrNumClasses)  
    arrDiscAction = action.descretize(arrContAction)

"""

import sys
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)


class Action(object):

    def __init__(self, arrLower, arrUpper, arrNumClasses):
        """

        Parameters
        ----------
        arrLower : a 1D array of double
            a 1D arry containing lowver value for each dimension
        arrUpper : a 1D array of double
            a 1D arry containing upper value for each dimension
        arrNumClasses : a 1D array of int
            a 1D arry containing # of disretized classes in each dimension

        Returns
        -------
        None
        """

        self.arrLower = arrLower  # a list containing lower value for each dimension
        self.arrUpper = arrUpper  # a list containing upper value for each dimension

        self.arrNumClasses = arrNumClasses  # # of disretized classes in each dimension

        self.dim_action = 0  # dimension of the actions

        self.arrIntervals = []  # action interaval array (arrUpper-arrUpper)

        self.arrDiscAction = []  # discretized action array

        self.setDimAction()

        self.checkLowerUpperArray()

        self.setArrIntervals()

    def setDimAction(self):
        """ This function verifies and sets the dimension of action space.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        if (len(self.arrLower) != len(self.arrUpper)):
            print("ERROR: The dimension of the arrLower and arrUpper should be equal!")
            sys.exit()

        if (len(self.arrLower) != len(self.arrNumClasses)):
            print("ERROR: The dimension of the arrLower, arrUpper, "
                  "and arrNumClasses should be equal!")
            sys.exit()

        # set the dimension of the action array
        self.dim_action = len(self.arrNumClasses)

        logging.debug(f"dim_action: {self.dim_action}")

    def checkLowerUpperArray(self):
        """ This function validates the lower and upper arrays

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        for i in range(self.dim_action):
            if (self.arrLower[i]) > (self.arrUpper[i]):
                print("ERROR: arrLower[i] <= arrUpper[i] for all i!")
                sys.exit()

    def setArrIntervals(self):
        """ This function sets interval for each action dimension, arrIntervals

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.arrIntervals = np.zeros(self.dim_action)

        for i in range(self.dim_action):
            self.arrIntervals[i] = (
                self.arrUpper[i] - self.arrLower[i]) / self.arrNumClasses[i]

        logging.debug(f"arrIntervals: {self.arrIntervals}", )

    def descretize(self, arrContAction):
        """This function discreteize continuous action using discrete uniform distribution.

        Parameters
        ----------
        arrContAction : a 1D array of double
            a 1D arry of continuous action values

        Returns
        -------
        a 1D array of int
            a 1D arry of discritizecd action values (arrDiscAction)
        """
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
