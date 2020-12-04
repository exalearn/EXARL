# © (or copyright) 2020. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
# National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
# Department of Energy/National Nuclear Security Administration. All rights in the program are
# reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
# 1;95;0c Security Administration. The Government is granted for itself and others acting on its behalf a
# nonexclusive, paid-up, irrevocable globalwide license in this material to reproduce, prepare
# derivative works, distribute copies to the public, perform publicly and display publicly, and
# to permit others to do so.


import exarl.mpi_settings as mpi_settings
import time
import gym
import envs
import agents
import workflows

from exarl.env_base import ExaEnv

import os
import csv
import sys
from mpi4py import MPI
import json

import utils.log as log
import utils.candleDriver as cd
logger = log.setup_logger(__name__, cd.run_params['log_level'])


from line_profiler import LineProfiler


class ExaLearner():

    def __init__(self, comm):
        # Global communicator
        self.global_comm = comm
        self.global_size = self.global_comm.size

        # Default training
        self.nepisodes = 1
        self.nsteps = 10
        self.results_dir = './results'  # Default dir, will be overridden by candle
        self.do_render = False

        self.process_per_env = int(cd.run_params['process_per_env'])
        self.action_type = cd.run_params['action_type']

        # Setup agent and environments
        self.agent_id = 'agents:' + cd.run_params['agent']
        self.env_id   = 'envs:' + cd.run_params['env']
        self.workflow_id = 'workflows:' + cd.run_params['workflow']

        # Sanity check before we actually allocate resources
        if self.global_size < self.process_per_env:
            sys.exit('EXARL::ERROR Not enough processes.')
        if (self.global_size - 1) % self.process_per_env != 0:
            sys.exit('EXARL::ERROR Uneven number of processes.')
        if self.global_size < 2 and self.workflow_id == 'workflows:async':
            print('\n################\nNot enough processes, running synchronous single learner ...\n################\n')
            self.workflow_id = 'workflows:' + 'sync'

        # Setup MPI
        mpi_settings.init(self.global_comm, self.process_per_env)
        self.agent, self.env, self.workflow = self.make()
        self.env.unwrapped.spec.max_episode_steps  = self.nsteps
        self.env.unwrapped._max_episode_steps = self.nsteps

        self.env.spec.max_episode_steps  = self.nsteps
        self.env._max_episode_steps = self.nsteps
        self.set_config()
        # self.env.set_env()
        self.env.reset()

    def make(self):
        # Create environment object
        env = gym.make(self.env_id).unwrapped
        env = ExaEnv(env)
        # Create agent object
        agent = None
        # Only agent_comm processes will create agents
        try:
            agent = agents.make(self.agent_id, env=env)
        except:
            logger.debug('Does not contain an agent')
        # Create workflow object
        workflow = workflows.make(self.workflow_id)
        return agent, env, workflow

    def set_training(self, nepisodes, nsteps):
        self.nepisodes = nepisodes
        self.nsteps    = nsteps
        if self.global_size > self.nepisodes:
            sys.exit(
                'EXARL::ERROR There is more resources allocated for the number of episodes.\nnprocs should be less than nepisodes.')
        self.env.unwrapped._max_episode_steps = self.nsteps
        self.env.unwrapped.spec.max_episode_steps  = self.nsteps
        self.env.spec.max_episode_steps  = self.nsteps
        self.env._max_episode_steps = self.nsteps

    # Use with CANDLE
    def set_config(self):
        params = cd.run_params
        self.set_training(int(params['n_episodes']), int(params['n_steps']))
        self.results_dir = params['output_dir']
        if not os.path.exists(self.results_dir):
            if (self.global_comm.rank == 0):
                os.makedirs(self.results_dir)

    def render_env(self):
        self.do_render = True

    def run(self):
        self.workflow.run(self)
