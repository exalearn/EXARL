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
import os
import gym
import numpy as np
from math import log10, floor
import exarl as erl
from exarl.utils.globals import ExaGlobals
from exarl.utils.histogram import histogram
import matplotlib.pyplot as plt

class ExaRoots(gym.Env):

    def __init__(self, random_start=False):
        super().__init__()

        # Load action step size, tolerance from config file
        self.action_step_size = ExaGlobals.lookup_params('action_step_size')
        self.tolerance = ExaGlobals.lookup_params('tolerance')
        self.random_start = random_start
        self.precision = np.float32

        # Set up the parabola
        self.coeff = ExaGlobals.lookup_params('coefficents')
        assert len(self.coeff) >= 3

        # https://en.wikipedia.org/wiki/Newton%27s_method
        self.f = lambda x: sum([j * (x ** i) for i, j in enumerate(self.coeff)])
        self.fprime = lambda x: sum([(i + 1) * j * (x ** i) for i, j in enumerate(self.coeff[1:])])
        self.newton = lambda x: x - (self.f(x) / self.fprime(x))

        # https://en.wikipedia.org/wiki/Geometrical_properties_of_polynomial_roots#:~:text=given%20explicitly.-,Lagrange%27s%20and%20Cauchy%27s%20bounds,-%5Bedit%5D
        self.high = max(1, sum([ abs(a / self.coeff[-1]) for a in self.coeff[:-1]]))
        # Lets only look for real roots
        self.low = 0

        # We get the biggest x and make all coeffs pos then multiply by -1
        self.worst_case_reward = -1 * sum([abs(j) * self.high ** i for i, j in enumerate(self.coeff)])
        self.debug("Worst case reward", self.worst_case_reward)
        
        # Make the observation space/state
        max_times_coeff = [abs(self.high * x) for x in self.coeff]
        self.observation_space = gym.spaces.Box(low=np.array([self.low] + max_times_coeff) * -1, high=np.array([self.high] + max_times_coeff), dtype=self.precision)
        self.reset()

        # Make action space
        self.action_space_type = ExaGlobals.lookup_params('action_space')
        if self.action_space_type == 'Discrete':
            quarter_actions = abs(floor(log10(self.action_step_size)))
            pos_encodings = [10 ** x for x in range(quarter_actions)] + [10 ** (-1 * x) for x in range(quarter_actions + 1) if x > 0]
            neg_encodings = [-1 * x for x in pos_encodings]
            self.encodings = neg_encodings + [0] + pos_encodings
            self.encodings.sort()
            self.action_space = gym.spaces.Discrete(len(self.encodings))
        else:
            self.action_space = gym.spaces.Box(low=np.array([-self.action_step_size]), high=np.array([self.action_step_size]), dtype=self.precision)

        # self.obsHisto = RootHistogram("obs", self.low, self.high)
        self.obsHisto = RootHistogram("obs", self.encodings[0], self.encodings[-1])
        self.actHisto = RootHistogram("act", self.encodings[0], self.encodings[-1])
        self.errHisto = RootHistogram("err", self.encodings[0], self.encodings[-1])
        self.error = []

    def debug(self, *args):
        if True:
            print(*args, flush=True)

    def log_steps(self, diff, reward):
        self.diff = diff
        self.total_reward += reward
        self.steps += 1
    
    def print_log(self):
        try:
            print(self.steps, self.diff, self.total_reward)
            self.errHisto.observe(self.diff)
        except:
            pass
        self.steps = 0
        self.diff = 0
        self.total_reward = 0

    def step(self, action):
        # JS: Update action
        if self.action_space_type == 'Discrete':
            to_add = self.encodings[action]
        else:
            to_add = action[0]

        # JS: Update newton approximate guess
        guess = self.state[0] + to_add

        self.obsHisto.observe(guess)
        self.actHisto.observe(to_add)

        # JS: Is the guess out of bounds
        # if guess < self.low or guess > self.high:
        #     print("HMMM", guess, self.low, self.high)
        #     return self.state, self.worst_case_reward * 100, True, {}

        self.debug("Guess:", guess, "Action:", action, to_add)
        self.debug("Parts:", self.f(guess), self.fprime(guess))

        # JS: Calc new approx answer
        result = self.newton(guess)
        self.debug("Result:", result)

        # JS: Look at difference
        diff = self.f(result)
        self.debug("      Diff:", diff)
        self.error.append(diff)
        # self.errHisto.observe(diff)

        # JS: Update approx answer
        self.state[0] = result
        self.state[1:] = [j * self.state[0] ** i for i, j in enumerate(self.coeff)]

        # JS: Calculate reward
        reward = 1.0 - diff**2.0
        self.log_steps(diff, reward)

        return self.state, reward, diff <= self.tolerance, {}

    def reset(self):
        self.print_log()
        self.state = np.zeros(len(self.coeff) + 1, dtype=self.precision)
        if self.random_start:
            self.state[0] = np.random.uniform(self.low, self.high)
        self.state[1:] = [j * self.state[0] ** i for i, j in enumerate(self.coeff)]
        return self.state

    def close(self):
        self.obsHisto.plot()
        self.actHisto.plot()
        self.errHisto.plot()
        results_dir = ExaGlobals.lookup_params('output_dir')
        plot_path = results_dir + '/Plots/Error.png'
        plt.plot(range(len(self.error)), self.error)
        # plt.yscale('log')
        plt.savefig(plot_path)
        plt.clf()

class ExaRootsRandomStart(ExaRoots):
    def __init__(self):
        super().__init__(random_start=True)

class RootHistogram:
    '''
    We want to set the space_size to the total number of exps we will run.
    This fits the case where we could int theory have 1 observation per across
    the whole space.  The bins less than that will let us save on space and do
    some interpolation for us!

    We can go higher on the space_size if we are interested in information across
    runs.  In this case maybe there are parts of the observation space not actually
    observed?!?!?! (SHOCK AND AWE)

    This conversation is all assuming a perfect hash which sha256 is not so...
    Also we did not use the internal hash because it shifts across runs 
    for "security"...
    '''
    def __init__(self, name, low, high, bins=100):
        self.name = name
        self.numBins = bins
        self.low = low
        self.high = high
        self.histogram = histogram(bins, minKey=low, maxKey=high)

    def observe(self, observation):        
        self.histogram.add(observation)

    def plot(self):
        results_dir = ExaGlobals.lookup_params('output_dir')
        if not os.path.exists(results_dir + '/Plots'):
            os.makedirs(results_dir + '/Plots')
        plot_path = results_dir + '/Plots/Histogram_' + self.name + '.png'
        keys, values = self.histogram.getEvenBins()
        plt.hist(keys, bins=self.numBins, weights=values)
        plt.ylabel("Occurrence")
        plt.xlabel("State Hash")
        plt.title("ExaRoots Observation State Occurrence")
        plt.savefig(plot_path)
        plt.clf()