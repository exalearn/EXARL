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

class ExaComm(ABC):
    global_comm = None
    agent_comm = None
    env_comm = None
    learner_comm = None
    num_learners = 1

    def __init__(self, comm, procs_per_env, num_learners):
        if ExaComm.global_comm is None:
            ExaComm.num_learners = num_learners
            ExaComm.global_comm = comm
            ExaComm.agent_comm, ExaComm.env_comm, ExaComm.learner_comm = comm.split(procs_per_env, num_learners)

    @abstractmethod
    def send(self, data, dest, pack=False):
        pass

    @abstractmethod
    def recv(self, data_type, data_count, source):
        pass

    @abstractmethod
    def bcast(self, data, root):
        pass

    @abstractmethod
    def barrier(self):
        pass

    @abstractmethod
    def reduce(self, arg, op, root):
        pass

    @abstractmethod
    def allreduce(self, arg, op):
        pass

    @abstractmethod
    def time(self):
        pass

    @abstractmethod
    def split(self, procs_per_env):
        pass

    def learner_rank():
        if ExaComm.agent_comm is not None:
            return 0
        return -1

    def is_learner():
        if ExaComm.agent_comm is not None:
            if ExaComm.agent_comm.rank < ExaComm.num_learners:
                return True
        return False

    def is_actor():
        if ExaComm.agent_comm is not None:
            if ExaComm.agent_comm.rank >= ExaComm.num_learners:
                return True
        return False

    def is_agent():
        return ExaComm.agent_comm is not None
