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
    def __init__(self, comm, length, fail_push, data=None, size=None, name=None):
        assert data is not None or size is not None, "ExaData: Must provided size or example data for MPI data structures"
        # JS: This is fine since it is a singleton for MPI based communicators.
        # Will throw an error if not using MPI base comm
        self.MPI = ExaComm.get_MPI()
        self.comm = comm
        self.length = length
        self.fail_push = fail_push

        if size is None:
            dataBytes = self.MPI.pickle.dumps(data)
            # JS: plus one for weird pickling errors...
            size = len(dataBytes) + 1

        self.dataSize = size
        self.name = name

    @abstractmethod
    def pop(self, rank, count=1):
        pass

    @abstractmethod
    def push(self, data, rank=None):
        pass
