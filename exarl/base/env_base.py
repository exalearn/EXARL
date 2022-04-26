# © (or copyright) 2020. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
# National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
# Department of Energy/National Nuclear Security Administration. All rights in the program are
# reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
# Security Administration. The Government is granted for itself and others acting on its behalf a
# nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
# derivative works, distribute copies to the public, perform publicly and display publicly, and
# to permit others to do so.
import os
from gym import Wrapper
from exarl.base.comm_base import ExaComm

class ExaEnv(Wrapper):
    def __init__(self, env, **kwargs):
        super(ExaEnv, self).__init__(env)
        self.env = env
        self.workflow_episode = 0
        self.workflow_step = 0

        # Use relative path not absolute
        self.base_dir = os.path.dirname(__file__)
        self.env_comm = ExaComm.env_comm

    def set_results_dir(self, results_dir):
        '''
        Default method to save environment specific information
        '''
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        # Top level directory
        self.results_dir = results_dir
