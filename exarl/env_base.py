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


import json, os, sys
from abc import ABC, abstractmethod

class ExaEnv(ABC):
    def __init__(self, **kwargs):        
        # Use relative path not absolute
        self.base_dir = os.path.dirname(__file__)
        print(self.base_dir)
        
    def set_config(self, env_data):
        self.env_data = env_data
        self.run_type = env_data['run_type']
         
       # Add any MPI parameters                                                                                       
        self.mpi_children_per_parent = int(env_data['mpi_children_per_parent']) if 'mpi_children_per_parent' in env_data.keys() else 0

        # Add any OMP parameters
        self.omp_thread = int(env_data['omp_thread']) if 'omp_thread' in env_data.keys() else 1   
        
        # Add any GPU parameters                                                                                                   

        # Executable                                                                                                               
        if(self.run_type == 'dynamic' and self.mpi_children_per_parent > 0):
            self.worker = (env_data['worker_app']).lower() if 'worker_app' in env_data.keys() else print('Specify worker app')
        else:
            self.worker = None

    def get_config(self):
        return self.env_data
        
    @abstractmethod
    def step(self, action):
        ''' 
        Required by all environment to be implemented by user 
        '''
        pass

    @abstractmethod
    def reset(self):
        ''' 
        Required by all environment to be implemented by user 
        '''
        pass

    @abstractmethod
    def set_env(self):
        '''
        Required by all environment to be implemented by user.
        This function is used to set all the hyper-parameters for the environment
        '''
        pass
