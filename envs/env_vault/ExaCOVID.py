# This material was prepared as an account of work sponsored by an agency of the
# United States Government.  Neither the United States Government nor the United
# States Department of Energy, nor Battelle, nor any of their employees, nor any
# jurisdiction or organization that has cooperated in the development of these
# materials, makes any warranty, express or implied, or assumes any legal
# liability or responsibility for the accuracy, completeness, or usefulness or
# any information, apparatus, product, software, or process disclosed, or
# represents that its use would not infringe privately owned rights. Reference
# herein to any specific commercial product, process, or service by trade name,
# trademark, manufacturer, or otherwise does not necessarily constitute or imply
# its endorsement, recommendation, or favoring by the United States Government
# or any agency thereof, or Battelle Memorial Institute. The views and opinions
# of authors expressed herein do not necessarily state or reflect those of the
# United States Government or any agency thereof.
#                 PACIFIC NORTHWEST NATIONAL LABORATORY
#                            operated by
#                             BATTELLE
#                             for the
#                   UNITED STATES DEPARTMENT OF ENERGY
#                    under Contract DE-AC05-76RL01830
from gym import spaces
import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(__file__) + '/pydemic/')
from pydemic.data.united_states import nyt, get_population, get_age_distribution
from pydemic import MitigationModel
from pydemic.models.seirpp import SimulationResult
from pydemic.models import SEIRPlusPlusSimulation
import gym


class ExaCOVID(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, **kwargs):
        super().__init__()
        """

        """
        # self.cfg_data = super.get_config()
        self.results_dir = ''

        ''' Initial key variable setup '''
        self.episodes = 0
        self.steps = 0
        self.initial_cases = 200
        self.icu_max = 1000
        self.model_dt = 0.05

        ''' Define the model time scale for each step '''
        self.time_init = 0         # [days] a month delay
        self.mitigation_dt = 1     # [days]
        self.mitigation_length = 7  # [day]

        ''' Mitigation factors is used as the action '''
        self.mitigation = None
        self.mitigation_times   = [0, self.mitigation_length]
        self.mitigation_factors = [0.5, 0.5]

        ''' Define the initial model parameters and distributions '''
        self.state = "Illinois"
        self.data = nyt(self.state)
        self.total_population = get_population(self.state)
        # print('self.total_population:{}'.format(self.total_population))
        self.age_distribution = get_age_distribution()
        # TODO: Use some initial time (Jan 1st, 2020)
        self.tspan = ('2020-01-01', '2020-02-01')
        self.date_max = pd.to_datetime('2020-06-01')
        self.t0 = 0
        self.tf = 8 * 30

        ''' Model default '''
        self.y0 = {}
        self.y0['infected'] = self.initial_cases * np.array(self.age_distribution)
        self.y0['susceptible'] = (
            self.total_population * np.array(self.age_distribution) - self.y0['infected']
        )
        # print('Total infected:{}'.format(self.y0['infected'][:].sum()))

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
        # print('Variables:{}'.format(self.state_variables))

        self.observation_space = spaces.Box(low=np.zeros(self.nstates),
                                            high=np.ones(self.nstates),
                                            dtype=np.float32)

        # Increase, Decrease, Don't change
        self.action_factors = [0, 0.01, 0.05, 0.1, -0.01, -0.05, -0.1]
        self.action_space = spaces.Discrete(7)

    def step(self, action):
        # print('step()')
        ''' Initial step variables '''
        done = False
        reward = 0
        info = ''
        self.steps += 1

        ''' Add new mitigation times '''
        self.mitigation_times.append(self.steps * self.mitigation_length)
        self.mitigation_times.append(self.mitigation_times[-1] + self.mitigation_dt)

        ''' Added previous mitgation values '''
        self.mitigation_factors.append(self.mitigation_factors[-1])
        ''' Add new mitigation value   '''
        new_factor = self.mitigation_factors[-1] + self.action_factors[action]
        self.mitigation_factors.append(new_factor)

        ''' Out of bounds'''
        if self.mitigation_factors[-1] > 1:
            done = True
            reward = -99
            info = 'Out of bounds (upper)'

        if self.mitigation_factors[-1] < 0:
            done = True
            reward = -99
            info = 'Out of bounds (lower)'

        ''' Create mitigation model time span '''
        tspan_tmp0 = self.tspan[0]
        tspan_tmp1 = (pd.to_datetime(self.tspan[0]) + (self.steps + 1) * self.mitigation_length * pd.Timedelta('1D')).strftime('%Y-%m-%d')
        self.tspan = (tspan_tmp0, tspan_tmp1)
        # print('tspan:{}'.format(self.tspan))

        ''' Create mitigation model time span '''
        self.t0, self.tf = 0, self.steps * self.mitigation_length

        ''' New mitigation policy '''
        print('mitigation times:{}'.format(self.mitigation_times))
        print('mitigation factors:{}'.format(self.mitigation_factors))
        self.mitigation = MitigationModel(self.t0, self.tf, self.mitigation_times, self.mitigation_factors)

        ''' Run the model with update mitigation trace '''
        sim = SEIRPlusPlusSimulation(self.total_population, self.age_distribution,
                                     mitigation=self.mitigation, **self.parameters)

        self.result = sim(self.tspan, self.y0, self.model_dt)
        # for key in self.result.y.keys():
        #    print('{}: {}'.format(key, self.result.y[key].sum(axis=1)[-1]))
        total_icu = self.result.y['icu'].sum(axis=1)[-1]
        if total_icu > self.icu_max:
            reward = -499
            done = True
            info = 'Exceeded the infection capacity'

        # Calculate the reward
        if done != True:
            reward = total_icu / (self.icu_max + 1)

        if pd.to_datetime(self.tspan[1]) >= self.date_max:
            done = True
            info = 'Reached the max date'

        # Convert dict to state array
        next_state = np.array([self.result.y[key][:][-1].sum() for key in self.state_variables])
        next_state /= self.total_population
        ##
        if self.steps > 1:
            self.render()
        return next_state, reward, done, info

    def reset(self):
        self.episodes += 1
        self.steps     = 0
        self.mitigation_times   = [0, self.mitigation_dt]
        self.mitigation_factors = [0.5, 0.5]

        self.total_population = get_population(self.state)
        # print('self.total_population:{}'.format(self.total_population))
        ##
        t0, tf = 0, self.mitigation_length  # TODO: What range should consider ?? ##
        mitigation = MitigationModel(t0, tf, self.mitigation_times, self.mitigation_factors)

        sim = SEIRPlusPlusSimulation(self.total_population, self.age_distribution,
                                     mitigation=mitigation, **self.parameters)

        tspan_tmp0 = '2020-01-01'
        tspan_tmp1 = (pd.to_datetime(tspan_tmp0) + self.mitigation_length * pd.Timedelta('1D')).strftime('%Y-%m-%d')
        self.tspan = (tspan_tmp0, tspan_tmp1)
        self.result = sim(self.tspan, self.y0, self.model_dt)
        # print('variables:{}'.format(self.result.y.keys()))
        next_state = np.array([self.result.y[key][-1][-1] for key in self.state_variables])
        next_state /= self.total_population

        return next_state

    def render(self):
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        plt.rcParams['font.family'] = [u'serif']
        plt.rcParams['font.size'] = 16
        fig, ax = plt.subplots(2, figsize=(18, 12))

        ''' Migigation strategy '''
        filename  = self.results_dir + 'covid_render_episode{}'.format(self.episodes)
        _t = np.linspace(self.t0, self.tf, 1000)
        ax[0].plot(_t, self.mitigation(_t))

        ''' Results '''
        # filename  = self.results_dir + 'sim_episode{}'.format(self.episodes)
        plot_compartments = [
            'icu',
            'infected',
            'positive',
            'all_dead',
            'hospitalized',
        ]
        # fig, ax = plt.subplots(figsize=(10, 6))
        for name in plot_compartments:
            # print(result.y[name].shape)
            ax[1].plot(self.result.t,
                       (self.result.y[name].sum(axis=1)),
                       label=name)

        ax[1].plot()
        # plot on y log scale
        ax[1].set_yscale('log')
        ax[1].set_ylim(ymin=0.8)

        # plot x axis as dates
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        # fig.autofmt_xdate()

        # create legend
        ax[1].legend(loc='center left', bbox_to_anchor=(1, .5))
        ax[1].set_xlabel('time')
        ax[1].set_ylabel('count (persons)')

        plt.savefig(filename)

        return 0
