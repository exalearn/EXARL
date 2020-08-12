# © (or copyright) 2020. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
# National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
# Department of Energy/National Nuclear Security Administration. All rights in the program are
# reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
#1;95;0c Security Administration. The Government is granted for itself and others acting on its behalf a
# nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
# derivative works, distribute copies to the public, perform publicly and display publicly, and
# to permit others to do so.


import time
import gym
import envs
import agents

from exarl.env_base import ExaEnv

import os
import csv
import sys
from mpi4py import MPI
import json

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RL-Logger')
logger.setLevel(logging.INFO)

import exarl.mpi_settings as mpi_settings

class ExaLearner():

    def __init__(self, run_params):
        # World communicator
        self.world_comm = MPI.COMM_WORLD
        world_rank = self.world_comm.rank
        self.world_size = self.world_comm.size
        
        # Default training
        self.nepisodes = 1
        self.nsteps = 10
        self.results_dir = './results' # Default dir, will be overridden by candle 
        self.do_render = False

        self.learner_type = run_params['learner_type']
        self.process_per_env = int(run_params['process_per_env'])

        ## Sanity check
        if self.world_size < self.process_per_env:
            sys.exit('Not enough processes.')
        if self.world_size % self.process_per_env != 0:
            sys.exit('Uneven number of processes.')
        if self.world_size < 2 and self.learner_type == 'async':
            print('\n################\nNot enough processes, running synchronous single learner ...\n################\n')

        ## Setup MPI
        mpi_settings.init(self.process_per_env)
        # Setup agent and environments
        agent_id = 'agents:'+run_params['agent']
        env_id   = 'envs:'+run_params['env']
        self.agent_id = agent_id
        self.env_id   = env_id
        self.agent, self.env = self.make(run_params)
        self.env.unwrapped.spec.max_episode_steps  = self.nsteps
        self.env.unwrapped._max_episode_steps = self.nsteps
        
        self.env.spec.max_episode_steps  = self.nsteps
        self.env._max_episode_steps = self.nsteps
        ##
        self.set_config(run_params)
        self.env.set_env()
        self.env.reset()
        
        
    def make(self,run_params):
        # Create environment object
        env = gym.make(self.env_id).unwrapped
        env = ExaEnv(env,run_params)
        agent = agents.make(self.agent_id, env=env)
 
        return agent, env


    def set_training(self,nepisodes,nsteps):
        self.nepisodes = nepisodes
        self.nsteps    = nsteps
        self.env.unwrapped._max_episode_steps = self.nsteps
        self.env.unwrapped.spec.max_episode_steps  = self.nsteps
        self.env.spec.max_episode_steps  = self.nsteps
        self.env._max_episode_steps = self.nsteps

    # Use with CANDLE
    def set_config(self, params):
        self.set_training(int(params['n_episodes']), int(params['n_steps']))
        # set the agent up
        self.agent.set_config(params)
        self.env.set_config(params)
        self.results_dir = params['output_dir']
        if not os.path.exists(self.results_dir):
            if (self.world_comm.rank == 0):
                os.makedirs(self.results_dir)

    def render_env(self):
        self.do_render=True
 
    def run(self, run_type):
        if self.agent!=None:
            self.agent.set_agent()

        if self.env!=None:
            self.env.set_env()

        # TODO add self.omp_num_threads as a param, override
        # with OMP_NUM_THREADS
        #os.environ['OMP_NUM_THREADS']='{:d}'.format(self.omp_num_threads)
        if self.learner_type == 'seed':
            from exarl.exa_seed import run_seed
            run_seed(self, mpi_settings.agent_comm)

        if self.learner_type == 'async' and self.world_size >= 2:
            from exarl.exa_async_learner import run_async_learner
            run_async_learner(self, mpi_settings.agent_comm)
        
        else:
            from exarl.exa_single_learner import run_single_learner
            run_single_learner(self, mpi_settings.agent_comm)
