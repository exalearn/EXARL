import gym
from gym import spaces
import time
import sys
import json
import exarl as erl
import numpy as np
import pandas as pd
import os, sys
## Load pydemic module ##
sys.path.append(os.path.dirname(__file__)+'/pydemic/')
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

        ##
        state = "Illinois"
        from pydemic.data.united_states import nyt, get_population, get_age_distribution
        self.data = nyt(state)
        self.total_population = get_population(state)
        self.age_distribution = get_age_distribution()

        self.tspan = ('2020-02-15', '2020-05-30')

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
        t0, tf = 50, 140
        times = [70, 80]
        factors = [1, .48]
        self.mitigation = MitigationModel(t0, tf, times, factors)

        ## Define state and action spaces
        self.policies_len  =1 ## Only using workfromhome policy for now
        self.workfraction_step =0.1
        self.structure_len = 1
        self.observation_space = spaces.Box(low=np.append(np.zeros(self.structure_len),[0.004]), high=np.append(np.ones(self.structure_len)*350,[0.012]),dtype=np.float32)

        ## Increase, Decrease, Don't change
        self.action_space = spaces.Discrete(3) 

        ## Create model parameter ##
        self.model_parameter = {}

    def step(self, action):
        ## Initial step variable ##
        done   = False
        reward = 0
        info   = ''
        
        # ## Apply discrete actions
        # if action==1:
        #     self.model_parameter['workfromhome']+=self.workfraction_step
        #     if self.model_parameter['workfromhome']>self.observation_space.high[0]:
        #         self.model_parameter['workfromhome']-=self.workfraction_step
        #         done = True
        #         return self._getState(),reward,done, {}
        #
        # elif action==2:
        #     self.model_parameter['workfromhome']-=self.workfraction_step
        #     if self.model_parameter['workfromhome']<self.observation_space.high[0]:
        #         self.model_parameter['workfromhome']+=self.workfraction_step
        #         done = True
        #         return self._getState(),reward,done, {}

        ## Make input file ##
            
        ## Run model state ##
        sim = SEIRPlusPlusSimulation(self.total_population, self.age_distribution,
                                     mitigation=self.mitigation, **self.parameters)
        
        ## Calculate the number of infected ##
        #reward = self.__get_reward(filename)
 
        
        return next_state, reward, done, info
    
    def __get_reward(self,filename):
        ''' Description:
             Calculates the total number of symptomatic people 
        '''
        reward        = 0
        total_people  = 0
        total_healthy = 0
        total_symp    = 0
        f = open(filename, "r")
        for x in f:
            if 'People:' in x:
                total_people = int(x.split()[-1])
            if 'Total symptomatic individuals by age:' in x:
                total_symp = sum( int(x) for x in ((x.split())[-1].split(','))[0:-1])

        total_healthy = total_people - total_symp
        if total_people!=0 and total_healthy>0:
            reward = total_healthy/total_people
            
        return reward
    
    def reset(self):
        return 0
    
    def render(self):
        return 0
