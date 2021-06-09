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

import sys
import os
from abc import ABC, abstractmethod
import numpy as np

class ExaData(ABC):
    def __init__(self, dataType, size, comm_size=1, max_model_lag=None):
        self.dataType = dataType
        self.dataSize = size
        self.comm_size = comm_size
        self.max_model_lag = max_model_lag

    @abstractmethod
    def pop(self, data, size=1):
        pass

    @abstractmethod
    def push(self, data):
        pass

    def get_data(self, learner_counter):
        batch_data = None
        actor_counter = -1
        while True:
            actor_idx = 0
            if self.comm_size > 1:
                actor_idx = np.random.randint(low=1, high=self.comm_size, size=1)[0]
            batch_data = self.pop(actor_idx)
            if batch_data:
                batch_data, actor_counter = batch_data
                if self.max_model_lag is None or learner_counter - actor_counter <= self.max_model_lag:
                    break
        return batch_data, actor_idx, actor_counter