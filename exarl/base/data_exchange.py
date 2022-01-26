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
from exarl.base.comm_base import ExaComm
# from exarl.utils.introspect import introspectTrace

class ExaData(ABC):
    def __init__(self, dataType, size, comm_size=1, max_model_lag=None, name=None):
        self.dataType = dataType
        self.dataSize = size
        if max_model_lag == "none":
            max_model_lag = None
        self.max_model_lag = max_model_lag
        self.name = name

    @abstractmethod
    def pop(self, rank, count=1):
        pass

    @abstractmethod
    def push(self, data, rank=None):
        pass

    # TODO: Think about low and high as parameters
    def get_data(self, learner_counter, low, high):
        actor_idx = np.random.randint(low=low, high=high, size=1)[0]
        batch_data = self.pop(actor_idx)
        if batch_data:
            batch_data, actor_counter = batch_data
            if self.max_model_lag is None or learner_counter - actor_counter <= self.max_model_lag:
                return batch_data, actor_idx, actor_counter
        return None, -1, -1

    # def get_data(self, learner_counter, low, high, attempts=None):
    #     batch_data = None
    #     actor_counter = -1
    #     actor_idx = 0
    #     attempt = 0
    #     while attempts is None or attempt < attempts:
    #         actor_idx = 0
    #         if self.comm_size > 1:
    #             actor_idx = np.random.randint(low=low, high=high, size=1)[0]
    #         batch_data = self.pop(actor_idx)
    #         if batch_data:
    #             batch_data, actor_counter = batch_data
    #             if self.max_model_lag is None or learner_counter - actor_counter <= self.max_model_lag:
    #                 break
    #         attempt += 1
    #     return batch_data, actor_idx, actor_counter
