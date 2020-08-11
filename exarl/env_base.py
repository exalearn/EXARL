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


import json, os, sys, gym, time
#from abc import ABC, abstractmethod
from gym import Wrapper
from mpi4py import MPI
import exarl.mpi_settings as mpi_settings

class ExaEnv(Wrapper):
    def __init__(self, env, **kwargs):  
        super(ExaEnv, self).__init__(env)     
        # Use relative path not absolute
        self.base_dir = os.path.dirname(__file__)
        print(self.base_dir)
        #self.env = env
        self.env_comm = mpi_settings.env_comm
        
        self.env_cfg = os.path.join(self.base_dir, '../envs/env_vault/env_cfg/env_setup.json')
        for key, value in kwargs.items():
            if key == 'env_cfg':
                self.env_cfg = value

        with open(self.env_cfg) as json_file:
            env_data = json.load(json_file)

        ## TODO: Add any MPI parameters
        self.process_per_env = int(env_data['process_per_env']) if 'process_per_env' in env_data.keys() else 0

        ## TODO: Add any OMP parameters
        self.omp_thread = int(env_data['omp_thread']) if 'omp_thread' in env_data.keys() else 1

        ## TODO: Add any GPU parameters


        ## TODO: executable 
        if(self.process_per_env > 0):
            # defaults to running toy example of computing PI
            self.worker = (env_data['worker_app']).lower() if 'worker_app' in env_data.keys() else "envs/env_vault/cpi.py"
        else:
            self.worker = None
        
        self.env = env
        
    def set_results_dir(self,results_dir):
        ''' 
        Default method to save environment specific information 
        '''
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        ## Top level directory 
        self.results_dir=results_dir

    def set_config(self, env_data):
        self.env_data = env_data

    def get_config(self):
        return self.env_data
        
    def set_env(self):
        '''
        Use this function to set hyper-parameters, if any')
        '''
        env_data = self.get_config()
        for key in env_data:
            if key!="env":
                setattr(self.env, key, env_data[key])
