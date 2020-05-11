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


import json
import os
from abc import ABC, abstractmethod

import sys
file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..', 'candlelib'))
sys.path.append(lib_path)

import keras
import candle

class ExaAgent(ABC):

    def __init__(self, **kwargs):
        self.candle = candle  # make CANDLE functions accessible to all agents.
        self.agent_data = {}

    # Default method to set output directory
    def set_results_dir(self,results_dir):
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        # Top level directory 
        self.results_dir=results_dir

    # Default method to get output directory
    def get_results_dir(self):
        return self.results_dir

    # Default method to set arguments
    def set_config(self, agent_data):
        self.agent_data = agent_data

    # Default method to get arguments   
    def get_config(self):
        return self.agent_data

    @abstractmethod
    def get_weights(self):
        pass

    @abstractmethod
    def set_weights(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def action(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def monitor(self):
        pass
