# © (or copyright) 2020. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
# National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
# Department of Energy/National Nuclear Security Administration. All rights in the program are
# reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
# Security Administration. The Government is granted for itself and others acting on its behalf a
# nonexclusive, paid-up, irrevocable globalwide license in this material to reproduce, prepare
# derivative works, distribute copies to the public, perform publicly and display publicly, and
# to permit others to do so.


import exarl.mpi_settings as mpi_settings
import time
import gym
import exarl.envs
import exarl.agents
import exarl.workflows

from exarl.base.env_base import ExaEnv

import os
import csv
import sys
from mpi4py import MPI
import json

import exarl.utils.log as log
import exarl.utils.candleDriver as cd
logger = log.setup_logger(__name__, cd.run_params['log_level'])


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

        self.learner_procs = int(cd.run_params['learner_procs'])
        self.process_per_env = int(cd.run_params['process_per_env'])
        self.action_type = cd.run_params['action_type']

        # Setup agent and environments
        self.agent_id = 'exarl.agents:' + cd.run_params['agent']
        self.env_id   = 'exarl.envs:' + cd.run_params['env']
        self.workflow_id = 'exarl.workflows:' + cd.run_params['workflow']

        # Sanity check before we actually allocate resources
        if self.global_size < self.process_per_env:
            sys.exit('EXARL::ERROR Not enough processes.')
        if (self.global_size - self.learner_procs) % self.process_per_env != 0:
            sys.exit('EXARL::ERROR Uneven number of processes.')
        if self.global_size < 2 and self.workflow_id != 'exarl.workflows:sync':
            print('')
            print('_________________________________________________________________')
            print('Not enough processes, running synchronous single learner ...')
            print('_________________________________________________________________', flush=True)
            self.workflow_id = 'exarl.workflows:' + 'sync'

        # Setup MPI
        mpi_settings.init(self.global_comm, self.learner_procs, self.process_per_env)

        # Create agent, environment, and workflow
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
        if mpi_settings.is_learner():
            agent = exarl.agents.make(self.agent_id, env=env, is_learner=True)
        elif  "seed" not in self.workflow_id and mpi_settings.is_actor():
            agent = exarl.agents.make(self.agent_id, env=env, is_learner=False)
        else:
            logger.debug('Does not contain an agent')
        # Create workflow object
        workflow = exarl.workflows.make(self.workflow_id)
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
