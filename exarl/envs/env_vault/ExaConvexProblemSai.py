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
import gym
import time
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import json
import exarl as erl
# from envs.env_vault.computePI import computePI as cp
import exarl.utils.log as log
import exarl.utils.candleDriver as cd
logger = log.setup_logger(__name__, cd.run_params['log_level'])
import exarl.mpi_settings as mpi_settings


def computePI(N, new_comm):
    h = 1.0 / N
    s = 0.0
    rank = new_comm.rank
    size = new_comm.size
    for i in range(rank, N, size):
        x = h * (i + 0.5)
        s += 4.0 / (1.0 + x**2)
    return s * h


class ExaConvexProblemSai(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        self.env_comm = mpi_settings.env_comm
        self.a = 2
        self.b = 4
        self.c = 1
        self.f = lambda x: self.a*x**2 + self.b*x + self.c

        self.xmin = -(self.b/(2.0*self.a))
        self.ymin = self.f(self.xmin)

        self.time_steps = 100

	#setup the action and state space

        self.high = 2
        self.low = -(self.high)
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=np.array([self.low]),high=np.array([self.high]))

       #initial state
        #self.state = [random.uniform(self.low,self.high)]
        self.state = [0]

       #for plotting
       #self.x = linspace(self.low,self.high,100)
       #self.y = self.f(self.x)


    def step(self, action):
        action_step_size = 0.1
        if action == 1 :
            self.state[0] += action_step_size
        else:
            self.state[0] -= action_step_size

        self.time_steps -= 1

        y_pred = self.f(self.state[0])

        ydiff = abs(self.ymin-y_pred)

        # set the tolerance limit

        tol = 0.1

        reward = 1.0 - ydiff**2


        time.sleep(0)  # Delay in seconds

        rank = self.env_comm.rank
        if rank == 0:
            N = 100
        else:
            N = None

        N = self.env_comm.bcast(N, root=0)
        myPI = computePI(N,self.env_comm) # calls python function
        #logger.info('Computin PI Rank[%s] - Step:%s PI= %s' %
        #                                (str(self.env_comm.rank), str(100-self.time_steps), str(myPI)))
        #print("Debug: Env Rank: %s , PI: %s" %  (str(rank),str(myPI)))

        #myPI = cp.compute_pi(N, self.env_comm) # Calls C++ function
        PI = self.env_comm.reduce(myPI, op=MPI.SUM, root=0)

        #if self.env_comm.rank == 0:
        #    print(PI)  # Print PI for verification



        if ydiff <= tol :
            done = True
            return self.state,reward,done, {}

        if self.time_steps <= 0:
            done = True
        else:
            done = False

        return self.state,reward,done, {}


    def reset(self):
        # self.env._max_episode_steps=self._max_episode_steps
        # print('Max steps: %s' % str(self._max_episode_steps))
        #self.state = [random.uniform(self.low,self.high)]
        self.state = [0]
        self.time_steps = 100
        return self.state

    def render(self, mode='human', close=False):
        #return self.env.render()
        #plt.clf()
        #plt.title("Convex problem example")
        #plt.plot(self.x,self.y)
        #plt.plot(self.xmin,self.ymin,"g*",label="Minimum")
        #plt.plot(self.state,self.f(self.state),"r*",label="Current")
        #plt.legend()
        #plt.pause(0.01)
        return


    def set_env(self):
        print('Use this function to set hyper-parameters, if any')
