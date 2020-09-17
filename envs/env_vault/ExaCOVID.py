import gym
from gym import spaces
import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(__file__) + '/pydemic/')
from pydemic.models import SEIRPlusPlusSimulation
from pydemic.models.seirpp import SimulationResult
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
        # self.cfg_data = super.get_config()

        self.steps = 0
        self.infected_max = 1000

        ''' Mitigation factor is used as the action '''
        self.factor_init = 1
        self.factor_final = 1

        ''' Define the model time scale for each step '''
        self.time_init = 30  # [days] a month delay
        self.mitigation_dt = 5  # [days]
        self.time_final = self.time_init + self.mitigation_dt
        self.initial_cases = 100

        ''' Define the initial model parameters and distributions '''
        state = "Illinois"
        from pydemic.data.united_states import nyt, get_population, get_age_distribution
        self.data = nyt(state)
        self.total_population = get_population(state)
        self.age_distribution = get_age_distribution()
        self.tspan = ('2020-02-15', '2020-02-20')

        def to_days(date):
            if isinstance(date, str):
                date = pd.to_datetime(date)
            if isinstance(date, pd.Timestamp):
                days = (date - pd.to_datetime('2020-01-01')) / pd.Timedelta('1D')
            else:
                days = date
            return days

        start_time, end_time = [to_days(x) for x in self.tspan]
        self.dt = 0.05
        times = np.arange(start_time, end_time + self.dt, self.dt)
        n_steps = times.shape[0]

        y0 = {}
        y0['infected'] = self.initial_cases * np.array(self.age_distribution)
        y0['susceptible'] = (
                self.total_population * np.array(self.age_distribution) - y0['infected']
        )
        y0_all_t = {}
        for key in y0:
            y0_all_t[key] = np.zeros(y0[key].shape + (n_steps,))
            y0_all_t[key][..., 0] = y0[key]

        self.result = SimulationResult(times, y0_all_t)
        #print(self.result.y['infected'])
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

        self.state_variables = SEIRPlusPlusSimulation.increment_keys
        self.nstates = len(self.state_variables)

        self.observation_space = spaces.Box(low=np.zeros(self.nstates),
                                            high=np.ones(self.nstates) * self.total_population,
                                            dtype=np.float32)

        ## Increase, Decrease, Don't change
        self.action_space = spaces.Discrete(3)
        self.action_add = 0.01
        
    def step(self, action):
        print('step()')
        ''' Initial step variables '''
        done = False
        reward = 0
        info = ''

        self.time_init = self.time_final
        self.time_final = self.time_init + self.mitigation_dt
        self.factor_init = self.factor_final

        ''' Apply discrete actions '''
        if action == 1:
            self.factor_final = self.factor_final + action_add
        elif action == 2:
            self.factor_final = self.factor_final - action_add

        ''' Out of bounds'''
        if self.factor_final > 1:
            done = True
            reward = -999
            info = 'Out of bounds (upper)'

        if self.factor_final < 0:
            done = True
            reward = -999
            info = 'Out of bounds (lower)'

        ''' Create mitigation model '''
        t0, tf = 0, 12 * self.mitigation_dt  ## TODO: What range should consider ?? ##
        times = [self.time_init, self.time_final]  ## days from start (2020/1/1) -- to be defined by step counter
        factors = [self.factor_init, self.factor_final]  ## To be optimized
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
        self.result = sim(self.tspan, self.y0, .05)
        if self.result.y['infected'] > self.infected_max:
            reward = -999
            done = True
            info = 'Exceeded the infection capacity'

        ''' Calculate the reward '''
        if done != True:
            reward = self.result.y['infected'] / (self.infected_max + 1)

        ''' Convert dict to state array '''
        next_state = np.array([result.y[key][-1][-1] for key in state_variables])

        return next_state, reward, done, info

    def reset(self):
        self.steps = 0
        self.result.y['infected'] = self.initial_cases * np.array(self.age_distribution)
        self.result.y['susceptible'] = (
                self.total_population * np.array(self.age_distribution) - self.result.y['infected']
        )
        # next_state = np.array([self.result.y[key][-1][-1] for key in self.state_variables])

        return np.zeros(self.nstates)

    # def render(self):
    #    return 0
