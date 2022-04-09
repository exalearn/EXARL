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
from abc import ABC, abstractmethod

class ExaComm(ABC):
    global_comm = None
    agent_comm = None
    env_comm = None
    learner_comm = None
    num_learners = 1
    procs_per_env = 1

    def __init__(self, comm, procs_per_env, num_learners):
        if ExaComm.global_comm is None:
            ExaComm.num_learners = num_learners
            ExaComm.procs_per_env = procs_per_env
            ExaComm.global_comm = comm
            ExaComm.agent_comm, ExaComm.env_comm, ExaComm.learner_comm = comm.split(procs_per_env, num_learners)

    @abstractmethod
    def raw(self):
        pass

    @abstractmethod
    def send(self, data, dest, pack=False):
        pass

    @abstractmethod
    def recv(self, data, source=None):
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

    @staticmethod
    def is_learner():
        return ExaComm.learner_comm is not None

    @staticmethod
    def is_actor():
        return ExaComm.env_comm is not None

    @staticmethod
    def is_agent():
        return ExaComm.agent_comm is not None

    @staticmethod
    def reset():
        ExaComm.global_comm = None
        ExaComm.agent_comm = None
        ExaComm.env_comm = None
        ExaComm.learner_comm = None
        ExaComm.num_learners = 1
        ExaComm.procs_per_env = 1

    def get_MPI():
        return ExaComm.global_comm.MPI