import json, os, sys
from abc import ABC, abstractmethod

class ExaEnv(ABC):
    def __init__(self, **kwargs):

        ## Locationn to save results
        ## TODO: Need to add MPI subdirectories
        self.results_dir = ''
        
        ## TODO: Use relative path not absolute
        self.base_dir = os.path.dirname(__file__)
        print(self.base_dir)
        for key, value in kwargs.items():
            if key == 'env_cfg':
                self.env_cfg = value
            else:
                self.env_cfg = os.path.join(dirname, '../envs/env_vault/env_cfg/env_setup.json')

        with open(self.env_cfg) as json_file:
            env_data = json.load(json_file)

        ## TODO: Add any MPI parameters
        self.mpi_child_spawn_per_parent = int(env_data['mpi_child_spawn_per_parent']) if 'mpi_child_spawn_per_parent' in env_data.keys() else 0

        ## TODO: Add any OMP parameters
        self.omp_thread = int(env_data['omp_thread']) if 'omp_thread' in env_data.keys() else 1

        ## TODO: Add any GPU parameters


        ## TODO: exacutable 
        #if(self.num_child_per_parent > 0):
        #    # defaults to running toy example of computing PI
        #    self.worker = (env_data['worker_app']).lower() if 'worker_app' in env_data.keys() else "envs/env_vault/cpi.py"
        #else:
        #    self.worker = None

    def set_results_dir(self,results_dir):
        ''' 
        Default method to save environment specific information 
        '''
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        ## Top level directory 
        self.results_dir=results_dir
        
    @abstractmethod
    def step(self, action):
        ''' 
        Required by all ennvironment to be implemented by user 
        '''
        pass

    @abstractmethod
    def reset(self):
        ''' 
        Required by all ennvironment to be implemented by user 
        '''
        pass


