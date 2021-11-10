import time
import sys
import gym
import json
import exarl as erl
import numpy as np
from gym import spaces
from gym.spaces.utils import flatten
from gym.spaces.utils import unflatten
import exarl.utils.candleDriver as cd

class Marshmallow(gym.Env):
    """
    This environment is used to test an agents ability to explore.
    We present different types of underlying functions for the
    policy network to discover.  Each step, we are progressing
    the domain of the function by one i.e. f(t) where t is the
    current step within the episode.

    There are two types of functions (function_type):
        Polynomial
        Approximate unit function (step function)
            using to logistic functions (https://en.wikipedia.org/wiki/Logistic_function)

    Input (for functions):

        Polynomial:
            function - list of polynomial coefficents
            i.e. [1, 2, -4, .5] -> f(t) = 1*t^0 + 2*t^1 - 4*t^2 + .5*t^3

        Unit:
            L - Height of the step
            x0 - mid point of the rising edge
            x1 - mid point of the falling edge

        Pulse:
            L - List of heights of the step
            x0 - List of mid-points of the rising edge
            x1 - List of mid-points of the falling edge

    Observation space (gym.spaces.Box): 

        Polynomial: [x[0]*t^0, x[1]*t^1, ... x[n-1]*t^n-1]
            where x in the input function and t is the step

        Unit: [x0-t, x1-t] where x0 and x1 are input and 
            t is the step

        Pulse: [x0[0]-t, x0[1]-t, ... x0[n-1]-t, x1[0]-t, x1[1]-t, ... x1[n-1]-t] 
            where x0 and x1 are input and t is the step

    Action space:

        The action space can either be discrete or continueous
        in order to test different types of agents. To change
        set action_space to Box or Discrete in configuration.

        Discrete: 
            0 -> stops the episode
            1 -> Continues the episode

        Box:
            0 -> stops the episode
            0 < x <= 1 -> Scales the reward
    
    Reward:
        The reward is the value f(t) at step t

    Attributes
    ----------
    flat_fuct : list of floats
        Number used to set up observation space and store coeffiecents
    eval : function pointer
        The evaluation function used to generate the reward and
        next state for each type of function
    k : int
        Changes the slop of the logistic function used in unit function
    L : int or list of ints
        Max values of unit and pulse functions
    x0 : int or list of ints
        Mid-points of rising edges of unit and pulse functions
    x1 : int or list of ints
        Mid-points of falling edges of unit and pulse functions
    observation_space : gym.spaces.Box
        See observation space description
    action_space : gym.spaces
        See action space discription
    initial_state : gym.spaces.Box
        State to reset observation space back to
    current_step : int
        Current step within an episode
    
    Methods
    -------
    step(action)
        Takes a step by evaluating underlying function a current step index

    reset()
        Resets the observation space and step index

    poly(action_value)
        Takes the action as a single number (float or int) and
        evaulates the polynomial function (i.e. f(t) and 
        returns reward and next state

    unit(action_value)
        Takes the action as a single number (float or int) and
        evaulates the unit function (i.e. f(t) and 
        returns reward and next state
        
    """
    def __init__(self):
        super().__init__()        
        which = cd.lookup_params('function_type', default='poly')
        if which == "poly":
            self.flat_funct = cd.lookup_params('function', default=[0, 1])
            self.eval = self.poly
            init = np.zeros(num_dims)
        elif which == "unit":
            self.k = 50
            self.L = cd.lookup_params('L', default=.5)
            self.x0 = cd.lookup_params('x0', default=6)
            self.x1 = cd.lookup_params('x1', default=9)
            self.flat_funct = [-1*self.x0, -1*self.x1]
            self.eval = self.unit
            init = np.asarray(self.flat_funct)
        else:
            self.k = 50
            self.L = cd.lookup_params('L', default=[.5])
            self.x0 = cd.lookup_params('x0', default=[6])
            self.x1 = cd.lookup_params('x1', default=[9])
            self.flat_funct = [-1*x0 for x0 in self.x0] + [-1*x1 for x1 in self.x1]
            self.eval = self.pulse
            init = np.asarray(self.flat_funct)
            
        num_dims = len(self.flat_funct)
        high = np.array([np.Inf] * num_dims, dtype=np.float64)
        low = np.array([np.NINF] * num_dims, dtype=np.float64)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float64)
        
        spaceDict = {
            "Discrete": spaces.Discrete(1),
            "Box": spaces.Box(low=np.array([0.0]), high=np.array([1.0]), dtype=np.float64),
        }
        self.action_space = spaceDict[cd.lookup_params('action_space', default='Discrete')]

        self.initial_state = unflatten(self.observation_space, init)
        self.current_step = 0

    def poly(self, action_value):
        coef = []
        for i, x in enumerate(self.flat_funct):
            coef.append(x * self.current_step ** i)
        reward = sum(coef) * action_value
        next_state = unflatten(self.observation_space, np.asarray(coef))
        return reward, next_state

    def unit(self, action_value):
        rise = self.L / (1 + np.exp(-1 * self.k *(self.current_step - self.x0)))
        fall = self.L / (1 + np.exp(-1 * self.k * (self.current_step - self.x1)))
        reward = action_value * (rise - fall)
        next_state = unflatten(self.observation_space, np.asarray([self.current_step - self.x0, self.current_step - self.x1]))
        return reward, next_state

    def pulse(self, action_value):
        reward = 0
        for x0, x1, L in zip(self.x0, self.x1, self.L):
            rise = L / (1 + np.exp(-1 * self.k *(self.current_step - x0)))
            fall = L / (1 + np.exp(-1 * self.k * (self.current_step - x1)))
            reward += rise - fall
        reward *= action_value
        next_state = [self.current_step - x0 for x0 in self.x0] + [self.current_step - x1 for x1 in self.x1]
        next_state = unflatten(self.observation_space, np.asarray(next_state))
        return reward, next_state

    def step(self, action):
        self.current_step += 1
        action_value = flatten(self.action_space, action)[0]
        done = action_value == 0
        reward, next_state = self.eval(action_value)
        # print(self.current_step, reward)
        return self.initial_state, reward, done, {}

    def reset(self):
        self.current_step = 0
        return self.initial_state
