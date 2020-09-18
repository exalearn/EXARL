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
from pydemic.data.united_states import nyt, get_population, get_age_distribution


class ExaCOVID(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, **kwargs):
        super().__init__()
        """ 
        
        """
        # self.cfg_data = super.get_config()

        self.steps = 0
        self.initial_cases = 100
        self.infected_max = 10000
        self.dt=0.05
        ''' Mitigation factor is used as the action '''
        self.factor_init = 0.5
        self.factor_final = 0.5

        ''' Define the model time scale for each step '''
        self.time_init = 0  # [days] a month delay
        self.mitigation_dt = 1  # [days]
        self.time_final = self.time_init + self.mitigation_dt
        self.mitigation_length = 5 # [day]

        ''' Define the initial model parameters and distributions '''
        self.state = "Illinois"
        self.data = nyt(self.state)
        self.total_population = get_population(self.state)
        print('self.total_population:{}'.format(self.total_population))
        self.age_distribution = get_age_distribution()
        ## TODO: Use some initial time (Jan 1st, 2020)
        self.tspan = ('2020-01-01', '2020-02-01')

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
        print('Variables:{}'.format(self.state_variables))
        #self.reset()

        self.observation_space = spaces.Box(low=np.zeros(self.nstates),
                                            high=np.ones(self.nstates) * self.total_population,
                                            dtype=np.float32)

        ## Increase, Decrease, Don't change
        self.action_space = spaces.Discrete(3)
        self.action_add = 0.05
        
    def step(self, action):
        print('step()')
        ''' Initial step variables '''
        done = False
        reward = 0
        info = ''

        #self.time_init = self.time_final
        #self.time_final = self.time_init + self.mitigation_dt
        self.factor_init = self.factor_final

        ''' Apply discrete actions '''
        if action == 1:
            self.factor_final = self.factor_final + self.action_add
        elif action == 2:
            self.factor_final = self.factor_final - self.action_add

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
        ## TODO: Hacking the timestamp/MitigationModel because SEIR model doesn't properly support MDP approach
        tspan_tmp0 = self.tspan[0]
        tspan_tmp1 =  (pd.to_datetime(self.tspan[0])+2*self.mitigation_length*pd.Timedelta('1D')).strftime('%Y-%m-%d')
        self.tspan = (tspan_tmp0,tspan_tmp1)
        print('tspan:{}'.format(self.tspan))
        t0, tf = 0, self.mitigation_length  ## TODO: What range should consider ?? ##
        times = [self.time_init, self.time_final]  ## days from start (2020/1/1) -- to be defined by step counter
        print('times:{}'.format(times))
        factors = [self.factor_init, self.factor_final]  ## To be optimized
        print('factors:{}'.format(factors))
        mitigation = MitigationModel(t0, tf, times, factors)
        _t = np.linspace(t0, tf, 10)
        print('mitigation{}'.format(mitigation(_t)))
        ## Run model state ##
        # TODO: Update the total population
        sim = SEIRPlusPlusSimulation(self.total_population, self.age_distribution,
                                     mitigation=mitigation, **self.parameters)

        ''' Run simulation and results (dict) '''
        self.total_population -= self.result.y['all_dead'].sum(axis=1)[-1]

        #self.y0 = {}
        #for key in self.result.y.keys():
        #    self.y0[key] = self.result.y[key][:][-1]
        #    print('y0{}: {}'.format(key, self.result.y[key].sum(axis=1)[-1]))
        # infected_per_age = self.result.y['infected'][:][-1]
        # total_infected = infected_per_age.sum()
        # self.y0 = {}
        # self.y0['infected'] = infected_per_age # .sum() * np.array(self.age_distribution)
        # ## TODO: Need to update the total population
        # self.y0['susceptible'] = (
        #         self.total_population * np.array(self.age_distribution) - self.y0['infected']
        # )
        ## TODO: Need to create a new tspan
        self.result = sim(self.tspan, self.y0, self.dt)
        #print(f'Total infected per age:{infected_per_age}')
        #print('Total infected:{}'.format(total_infected))
        total_infected = self.result.y['infected'].sum(axis=1)[-1]
        for key in self.result.y.keys():
            print('result.y{}: {}'.format(key, self.result.y[key].sum(axis=1)[-1]))
        if total_infected > self.infected_max:
            reward = -999
            done = True
            info = 'Exceeded the infection capacity'

        ''' Calculate the reward '''
        if done != True:
            reward = total_infected / (self.infected_max + 1)

        ''' Convert dict to state array '''
        next_state = np.array([self.result.y[key][:][-1].sum() for key in self.state_variables])
        self.steps+=1
        return next_state, reward, done, info

    def reset(self):
        self.steps = 0
        self.total_population = get_population(self.state)
        print('self.total_population:{}'.format(self.total_population))
        ##
        ''' Create mitigation model '''
        t0, tf = 0, self.mitigation_length  ## TODO: What range should consider ?? ##
        times = [self.time_init, self.time_final]  ## days from start (2020/1/1) -- to be defined by step counter
        factors = [self.factor_init, self.factor_final]  ## To be optimized
        mitigation = MitigationModel(t0, tf, times, factors)
        _t = np.linspace(t0, tf, 10)
        print('mitigation{}'.format(mitigation(_t)))

        sim = SEIRPlusPlusSimulation(self.total_population, self.age_distribution,
                                     mitigation=mitigation, **self.parameters)

        self.y0 = {}
        self.y0['infected'] = self.initial_cases * np.array(self.age_distribution)
        self.y0['susceptible'] = (
                self.total_population * np.array(self.age_distribution) - self.y0['infected']
        )
        print('Total infected:{}'.format(self.y0['infected'][:].sum()))

        tspan_tmp0 = '2020-01-01'
        tspan_tmp1 =  (pd.to_datetime(tspan_tmp0)+self.mitigation_length*pd.Timedelta('1D')).strftime('%Y-%m-%d')
        self.tspan = (tspan_tmp0,tspan_tmp1)
        self.result = sim(self.tspan, self.y0, self.dt)
        print('variables:{}'.format(self.result.y.keys()))
        next_state = np.array([self.result.y[key][-1][-1] for key in self.state_variables])
        total_infected = self.result.y['infected'].sum(axis=1)[-1]
        #for key in self.result.y.keys():
        #    print('{}: {}'.format(key, self.result.y[key].sum(axis=1)[-1]))

        return next_state

    # def render(self):
    #    return 0
