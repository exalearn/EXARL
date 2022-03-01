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
# This material was prepared as an account of work sponsored by an agency of the
# United States Government.  Neither the United States Government nor the United
# States Department of Energy, nor Battelle, nor any of their employees, nor any
# jurisdiction or organization that has cooperated in the development of these
# materials, makes any warranty, express or implied, or assumes any legal
# liability or responsibility for the accuracy, completeness, or usefulness or
# any information, apparatus, product, software, or process disclosed, or
# represents that its use would not infringe privately owned rights. Reference
# herein to any specific commercial product, process, or service by trade name,
# trademark, manufacturer, or otherwise does not necessarily constitute or imply
# its endorsement, recommendation, or favoring by the United States Government
# or any agency thereof, or Battelle Memorial Institute. The views and opinions
# of authors expressed herein do not necessarily state or reflect those of the
# United States Government or any agency thereof.
#                 PACIFIC NORTHWEST NATIONAL LABORATORY
#                            operated by
#                             BATTELLE
#                             for the
#                   UNITED STATES DEPARTMENT OF ENERGY
#                    under Contract DE-AC05-76RL01830
import os
import sys
import pickle
from abc import ABC, abstractmethod


file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..', 'candlelib'))
sys.path.append(lib_path)

class ExaAgent(ABC):
    """Agent base class: Inherits from abstract base class for mandating
    functionality (pure virtual functions).

    """

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def get_weights(self):
        """get target model weights
    """
        pass

    @abstractmethod
    def set_weights(self, weights):
        """set target model weights
        """
        pass

    @abstractmethod
    def train(self, batch):
        """train the agent
        """
        pass

    @abstractmethod
    def target_train(self):
        pass

    @abstractmethod
    def action(self, state):
        """next action based on current state
        """
        pass

    # @abstractmethod
    # def load(self, filename):
    #     """load weights
    #     """
    #     pass

    # @abstractmethod
    # def save(self, filename):
    #     """save weights
    #     """
    #     pass

    @abstractmethod
    def has_data(self):
        """return true if agent has experiences from simulation
        """
        pass

    @abstractmethod
    def generate_data(self):
        pass

    @abstractmethod
    def remember(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def set_priorities(self, indices, loss):
        pass

    def load(self, filename):
        weights = None
        with open(filename, "rb") as f:
            weights = pickle.load(f)
        if weights is not None:
            print("Loading from: ", filename)
            self.set_weights(weights)
        else:
            print("Failed loading weights from:", filename)

    def save(self, filename):
        weights = self.get_weights()
        with open(filename, "wb") as f:
            pickle.dump(weights, f)