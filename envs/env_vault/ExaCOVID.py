import gym
from gym import spaces
import time
import sys
import json
import exarl as erl
import numpy as np
import pandas as pd

from pydemic.models import SEIRPlusPlusSimulation
from pydemic import MitigationModel

class ExaCOVID(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, **kwargs):
        super().__init__()
        """ 
        Description:
           Environment used to run the CORVID code

        Model source code:
           https://github.com/balewski/exaLearnEpi/tree/master/corvid_march

        Observation states map directly to the model input parameters that are being changed by the actions
           - 
           - 

        Action space is discrete: 
           - Increase 
           - 

        Reward is the number of people infected (??)

        """
        #self.cfg_data = super.get_config()

        self.infected_max=1000

        ''' Mitigation factor is used as the action '''
        self.factor_init = 1
        self.factor_final = 1

        ''' Define the model time scale for each step '''
        self.time_init = 30 # [days] a month delay
        self.mitigation_dt = 7 # [days]
        self.time_final = self.time_init+self.mitigation_dt

        ''' Define the initial model parameters and distributions '''
        state = "Illinois"
        from pydemic.data.united_states import nyt, get_population, get_age_distribution
        self.data = nyt(state)
        self.total_population = get_population(state)
        self.age_distribution = get_age_distribution()
        self.tspan = ('2020-02-15', '2020-05-30')

        self.initial_cases = 100

        from pydemic.distributions import GammaDistribution

        self.parameters = dict(
            ifr=.003,
            r0=2.3,
            serial_dist=GammaDistribution(mean=4, std=3.25),
            seasonal_forcing_amp=.1,
            peak_day=15,
            incubation_dist=GammaDistribution(5.5, 2),
            p_symptomatic=np.array([0.057, 0.054, 0.294, 0.668, 0.614, 0.83,
                                    0.99, 0.995, 0.999]),
            # p_positive=1.5,
            hospitalized_dist=GammaDistribution(6.5, 1.6),
            p_hospitalized=np.array([0.001, 0.003, 0.012, 0.032, 0.049, 0.102,
                                     0.166, 0.243, 0.273]),
            discharged_dist=GammaDistribution(9, 6),
            critical_dist=GammaDistribution(3, 1),
            p_critical=.9 * np.array([0.05, 0.05, 0.05, 0.05, 0.063, 0.122,
                                      0.274, 0.432, 0.709]),
            dead_dist=GammaDistribution(7.5, 5.),
            p_dead=1.2 * np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5]),
            recovered_dist=GammaDistribution(9, 2.2),
            all_dead_dist=GammaDistribution(3, 3),
            all_dead_multiplier=1.,
        )

        ''' Observation state based on the seirpp.py dict
            increment_keys = ('infected', 'dead', 'all_dead', 'positive',
                      'admitted_to_hospital', 'total_discharged')
        '''
        self.result = None
        self.result.y = {}
        self.result.y['infected'] = self.initial_cases * np.array(self.age_distribution)
        self.result.y['susceptible'] = (
                self.total_population * np.array(self.age_distribution) - self.result.y['infected']
        )
        self.observation_space = spaces.Box(low=np.append(np.zeros(6)),
                                            high=np.append(np.ones(6)*self.total_population),
                                            dtype=np.float32)

        ## Increase, Decrease, Don't change
        self.action_space = spaces.Discrete(3)
        self.action_add = 0.01

    def step(self, action):

        ''' Initial step variables '''
        done   = False
        reward = 0
        info   = ''

        self.time_init = self.time_final
        self.time_final = self.time_init+self.mitigation_dt
        self.factor_init = self.factor_final

        ''' Apply discrete actions '''
        if action==1:
            self.factor_final = self.factor_final + action_add
        elif action == 2:
            self.factor_final = self.factor_final - action_add

        ''' Out of bounds'''
        if self.factor_final>1:
            done = True
            reward = -99
            info = 'Out of bounds (upper)'

        if self.factor_final < 0:
            done = True
            reward = -99
            info = 'Out of bounds (lower)'

        ''' Create mitigation model '''
        t0, tf  = 0, 12*self.mitigation_dt ## TODO: What range should consider ?? ##
        times   = [self.time_init, self.time_final]  ## days from start (2020/1/1) -- to be defined by step counter
        factors = [self.factor_init, self.factor_final] ## To be optimized
        mitigation = MitigationModel(t0, tf, times, factors)

        ## Run model state ##

        sim = SEIRPlusPlusSimulation(self.total_population, self.age_distribution,
                                     mitigation=mitigation, **self.parameters)

        ''' Run simulation and results (dict) '''
        self.y0 = {}
        self.y0['infected'] = self.result.y['infected'] * np.array(self.age_distribution)
        ## TODO: Need to update the total population
        self.y0['susceptible'] = (
                self.total_population * np.array(self.age_distribution) - self.y0['infected']
        )
        self.result = self.sim(tspan, y0, .05)
        if self.result.y['infected']>self.infected_max:
            reward = -999
            done = True
            info = 'Exceeded the infection capacity'

        ''' Convert dict to state array '''
        return next_state, reward, done, info
    
    def __get_reward(self,filename):
        ''' Description:
             Calculates the total number of symptomatic people 
        '''

            
        return reward
    
    def reset(self):

        self.result.y['infected'] = self.initial_cases * np.array(self.age_distribution)
        self.result.y['susceptible'] = (
                self.total_population * np.array(self.age_distribution) - self.result.y['infected']
        )

        return 0
    
    def render(self):
        return 0
