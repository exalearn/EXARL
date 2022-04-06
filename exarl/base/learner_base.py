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
import os
import sys
import gym
import exarl.envs
import exarl.agents
import exarl.workflows
from exarl.network.simple_comm import ExaSimple
# from exarl.network.mpi_comm import ExaMPI
from exarl.network.typing import TypeUtils
from exarl.base.comm_base import ExaComm
from exarl.base.env_base import ExaEnv
from exarl.utils import log
import exarl.utils.candleDriver as cd
logger = log.setup_logger(__name__, cd.lookup_params('log_level', [3, 3]))

class ExaLearner:
    def __init__(self, comm=None):
        # Setup agent and environments
        agent_id = cd.run_params['agent']
        env_id = cd.run_params['env']
        workflow_id = cd.run_params['workflow']
        
        learner_procs = int(cd.run_params['learner_procs'])
        process_per_env = int(cd.run_params['process_per_env'])

        self.nepisodes = int(cd.run_params['n_episodes'])
        self.nsteps = int(cd.run_params['n_steps'])
        self.action_type = cd.run_params['action_type']
        self.results_dir = cd.run_params['output_dir']

        # Setup MPI Global communicator
        ExaSimple(comm, process_per_env, learner_procs)

        # Sanity check before we actually allocate resources
        

        self.create_output_dir()
        self.agent, self.env, self.workflow = self.make(agent_id, env_id, workflow_id)
        self.set_training()
        if ExaComm.is_actor():
            self.env.reset()

    def sanity_check(self, agent_id, env_id, workflow_id):
        global_size = ExaComm.global_comm.size
        if global_size > nepisodes:
            sys.exit("EXARL::ERROR More resources allocated for the number of episodes.\n" +
                     "Number of ranks should be less than the number of episodes.")
        if global_size < process_per_env:
            sys.exit('EXARL::ERROR Not enough processes.')
        if workflow_id == 'exarl.workflows:sync':
            if learner_procs > 1:
                sys.exit('EXARL::sync learner only works with single learner.')
            if global_size != process_per_env:
                sys.exit('EXARL::sync learner can only run one env group.')
        else:
            if (global_size - learner_procs) % process_per_env != 0:
                sys.exit('EXARL::ERROR Uneven number of processes.')
        if learner_procs > 1 and workflow_id != 'exarl.workflows:rma':
            print('')
            print('_________________________________________________________________')
            print('Multilearner is only supported in RMA, running rma workflow ...')
            print('_________________________________________________________________', flush=True)
            workflow_id = 'exarl.workflows:' + 'rma'
        if global_size < 2 and workflow_id != 'exarl.workflows:sync':
            print('')
            print('_________________________________________________________________')
            print('Not enough processes, running synchronous single learner ...')
            print('_________________________________________________________________', flush=True)
            workflow_id = 'exarl.workflows:' + 'sync'

    def make(self, agent_id, env_id, workflow_id):
        # Create environment object
        env = gym.make(env_id).unwrapped
        env = ExaEnv(env)

        # Only agent_comm processes will create agents
        agent = None
        if ExaComm.is_agent():
            agent = exarl.agents.make(agent_id, env=env, is_learner=ExaComm.is_learner())

        # Create workflow object
        workflow = exarl.workflows.make(workflow_id)
        return agent, env, workflow

    def set_training(self):
        self.env.unwrapped._max_episode_steps = self.nsteps
        self.env.unwrapped.spec.max_episode_steps = self.nsteps
        self.env.spec.max_episode_steps = self.nsteps
        self.env._max_episode_steps = self.nsteps

    def create_output_dir(self):
        if ExaComm.global_comm == 0:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)

    def run(self):
        self.workflow.run(self)
