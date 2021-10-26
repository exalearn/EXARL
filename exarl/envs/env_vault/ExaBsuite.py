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
import gym
import gym.spaces as spaces
import time
import numpy as np
import sys
import json
import exarl as erl
from typing import Any, Dict, Optional, Tuple, Union

from exarl.base.comm_base import ExaComm

import bsuite
from bsuite.utils import gym_wrapper

_GymTimestep = Tuple[np.ndarray, float, bool, Dict[str, Any]]

class ExaBsuite(gym.Env):
	def __init__(self) -> None:
		super().__init__()
		self.env_comm = ExaComm.env_comm
		# TODO: Might also want to account for bounded envs too (like catch).
		env = bsuite.load_from_id(bsuite_id='cartpole/0')
		self.env = gym_wrapper.GymFromDMEnv(env)
		self.action_space = self.env.action_space
		self.has_extra_dim = False
		if self.env.observation_space.shape[0] == 1:
			shape = self.env.observation_space.shape[1]
			self.observation_space = spaces.Box(-np.inf, np.inf, shape=[shape])
			self.has_extra_dim = True
		else:
			self.observation_space = self.env.observation_space
		
		print("action space: ", type(self.action_space))
		print("obs space: ", type(self.observation_space))
		print("action space size: ", self.action_space.n)
		print("obs space: ", self.observation_space)
		print("obs dtype: ", self.env.observation_space.dtype)

	def step(self, action) -> _GymTimestep:
		time.sleep(0)
		next_state, reward, done, info = self.env.step(action)
		if self.has_extra_dim:
			next_state = next_state[0]
		return next_state, reward, done, info

	def reset(self) -> np.ndarray:
		if self.has_extra_dim:
			return self.env.reset()[0]
		return self.env.reset()

	def render(self, mode: str = 'human') -> Union[np.ndarray, bool]:
		return self.env.render(mode)