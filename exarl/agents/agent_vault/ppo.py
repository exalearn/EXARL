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
import numpy as np
import tensorflow as tf
from copy import deepcopy

import exarl
from exarl.utils.globals import ExaGlobals
from exarl.agents.replay_buffers.buffer import Buffer
from exarl.utils.OUActionNoise import OUActionNoise
from exarl.agents.models.tf_model import Tensorflow_Model
from exarl.utils.introspect import introspectTrace

logger = ExaGlobals.setup_logger(__name__)

class PPO(exarl.ExaAgent):
    def __init__(self, env, is_learner):
        self.is_learner = is_learner

    @introspectTrace()
    def action(self, state):
        pass

    def remember(self, state, action, reward, next_state, done):
        pass

    def has_data(self):
        pass

    def generate_data(self):
        pass

    def train(self, batch):
        pass

    def train_return(self, args):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass

    