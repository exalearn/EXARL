from exarl.agents.replay_buffers.replay_buffer      import ReplayBuffer, SimpleBuffer
from exarl.agents.replay_buffers.prioritized_replay import PrioritizedReplayBuffer
from exarl.agents.replay_buffers.trajectory_buffer  import TrajectoryBuffer
from exarl.agents.replay_buffers.nStep_buffer       import nStepBuffer
from exarl.utils.globals import ExaGlobals

class Buffer:
    _builders = {"ReplayBuffer": ReplayBuffer,
                 "PrioritizedReplayBuffer": PrioritizedReplayBuffer,
                 "TrajectoryBuffer": TrajectoryBuffer,
                 "nStepBuffer": nStepBuffer,
                 "SimpleBuffer" : SimpleBuffer}

    _config_args = {"ReplayBuffer": [],
                    "PrioritizedReplayBuffer": [],
                    "TrajectoryBuffer": ['buffer_trajectory_length'],
                    "nStepBuffer":["horizon", "gamma"],
                    "SimpleBuffer" : [] }

    def create(key=None, **kwargs):
        # JS: Lookup which buffer if not passed
        if key is None:
            key = ExaGlobals.lookup_params('buffer')
        
        # JS: Ensure buffer is listed
        builder = Buffer._builders.get(key)
        if not builder:
            raise ValueError(key)
        # JS: Look to see if capacity was passed
        capacity = kwargs.pop("capacity", None)
        for config_arg in Buffer._config_args[key]:
            # JS: make sure the required kwargs exist or are looked up
            if config_arg not in kwargs:
                kwargs[config_arg] = ExaGlobals.lookup_params(config_arg)
        # JS: Lookup the capacity if it was not passed in
        if capacity is None:
            capacity = ExaGlobals.lookup_params('buffer_capacity')
        return builder(capacity, **kwargs)
