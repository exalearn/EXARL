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
from os import path
import numpy as np
import gym
import gym.spaces as spaces
from typing import Any, Dict, Optional, Tuple, Union, Sequence
from exarl.base.comm_base import ExaComm
from exarl.utils.globals import ExaGlobals

import bsuite
from bsuite.utils import gym_wrapper
from bsuite.logging.csv_logging import Logger as CSVLogger

_GymTimestep = Tuple[np.ndarray, float, bool, Dict[str, Any]]

# Inspired by https://github.com/deepmind/bsuite/blob/master/bsuite/utils/wrappers.py

class BsuiteWrapper(gym.Env):
    """Environment wrapper to track and log bsuite stats in ExaRL."""
    def __init__(self) -> None:
        super().__init__()
        self.env_comm = ExaComm.env_comm
        rank = ExaComm.agent_comm.rank
        bsuite_id = ExaGlobals.lookup_params("bsuite_id")
        seed_number = ExaGlobals.lookup_params("seed_number")
        env_name = bsuite_id + "/" + seed_number
        print("Loading", env_name)

        # Let self.raw_env be of class dm_env.Environment.
        # Then return gym-like outputs for step, reset methods.
        self.raw_env = bsuite.load_from_id(bsuite_id=env_name)
        post_path = 'bsuite_results/' + "_".join([bsuite_id, str(seed_number), str(rank)])
        bsuite_res_path = path.join(ExaGlobals.lookup_params("output_dir"), post_path)
        self._logger = CSVLogger(bsuite_id=env_name, results_dir=bsuite_res_path, overwrite=True)

        self.env = gym_wrapper.GymFromDMEnv(self.raw_env)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        # Accumulating throughout experiment.
        self._steps = 0
        self.workflow_episode = 0
        self._total_return = 0.0

        # Most-recent-episode.
        self._episode_len = 0
        self._episode_return = 0.0

        self._log_by_step = False
        self._log_every = False

    def step(self, action) -> _GymTimestep:
        timestep = self.raw_env.step(action)
        self._track(timestep)
        next_state = timestep.observation
        reward = timestep.reward
        done = timestep.step_type.last()
        return next_state, reward, done, {}

    def reset(self) -> np.ndarray:
        timestep = self.raw_env.reset()
        self._track(timestep)
        return timestep.observation

    def _track(self, timestep):
        # Count transitions only.
        if not timestep.first():
            self._steps += 1
            self._episode_len += 1

        if timestep.last():
            self.workflow_episode += 1

        self._episode_return += timestep.reward or 0.0
        self._total_return += timestep.reward or 0.0

        # Log statistics periodically, either by step or by episode.
        if ExaComm.env_comm.rank == 0:
            if self._log_by_step:
                if _logarithmic_logging(self._steps) or self._log_every:
                    self._log_bsuite_data()

            elif timestep.last():
                if _logarithmic_logging(self.workflow_episode) or self._log_every:
                    self._log_bsuite_data()

        # Perform bookkeeping at the end of episodes.
        if timestep.last():
            self._episode_len = 0
            self._episode_return = 0.0

        if self.workflow_episode == self.raw_env.bsuite_num_episodes:
            self.flush()

    def _log_bsuite_data(self):
        """Log summary data for bsuite."""
        data = dict(
            # Accumulated data.
            steps=self._steps,
            episode=self.workflow_episode,
            total_return=self._total_return,
            # Most-recent-episode data.
            episode_len=self._episode_len,
            episode_return=self._episode_return,
        )
        # Environment-specific metadata used for scoring.
        data.update(self.raw_env.bsuite_info())
        self._logger.write(data)

    def flush(self):
        if hasattr(self._logger, 'flush'):
            self._logger.flush()

def _logarithmic_logging(episode: int,
                         ratios: Optional[Sequence[float]] = None) -> bool:
    """Returns `True` only at specific ratios of 10**exponent."""
    if ratios is None:
        ratios = [1., 1.2, 1.4, 1.7, 2., 2.5, 3., 4., 5., 6., 7., 8., 9., 10.]
    exponent = np.floor(np.log10(np.maximum(1, episode)))
    special_vals = [10**exponent * ratio for ratio in ratios]
    return any(episode == val for val in special_vals)
