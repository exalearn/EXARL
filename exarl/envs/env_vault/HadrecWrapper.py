import gym
from gym import spaces
import numpy as np
from exarl.base.comm_base import ExaComm
from exarl.utils.globals import ExaGlobals

class HadrecWrapper(gym.Env):
    high = np.array([1, 1, 1], dtype=np.float64)
    spaceDict = {
        "Discrete": spaces.Discrete(5),
        "Box_three": spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float64),
        "Box_seven": spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float64),
        "Box": spaces.Box(low=-high, high=high, dtype=np.float64),
        "MultiBinary": spaces.MultiBinary([2, 3]),
        "MultiDiscrete": spaces.MultiDiscrete([3, 2]),
        "Dict": spaces.Dict({
            "discrete": spaces.Discrete(100),
            "box": spaces.Box(low=-1, high=1, shape=(3, 4), dtype=np.float64),
            "multiBinary": spaces.MultiBinary([2, 3]),
            "multiDiscrete": spaces.MultiDiscrete([3, 2])
        })
    }
    

    def __init__(self):
        super().__init__()
        self.action_space = HadrecWrapper.spaceDict["Box_three"]
        self.observation_space = HadrecWrapper.spaceDict["Box_seven"]
        self.num_seeds = 1000
        self.seed = 0
        self.initial_state = []

        for i in range(self.num_seeds):
            self.initial_state.append((self.observation_space.sample()))

        if ExaComm.is_actor():
            self.initial_state = ExaComm.env_comm.bcast(self.initial_state, root=0)

        self.state = self.initial_state[self.seed]
        self.seed = (self.seed + 1) % self.num_seeds
        self.step_index = 0
        self.max_steps = ExaGlobals.lookup_params('n_steps')
        self.workflow_episode = 0
        self.Num_perturb = 2

    def step(self, action):
        if self.step_index < self.max_steps:
            self.state = self.observation_space.sample()
            self.step_index += 1
        done = self.step_index == self.max_steps
        done = False
        return self.state, 1, done, {}


    def reset(self):
        # This require that the Hadrec code reset to a fault specific secenario.
        # The workflow_episode couter is incremented by the sync learner every time 
        # the epsidode is completed.
        if self.workflow_episode % self.Num_perturb == 0 and self.workflow_episode >= self.Num_perturb:
            # Reset the environment with the new fault case
            self.state = self.initial_state[self.seed]
            self.seed = (self.seed + 1) % self.num_seeds
            print(f"Wrapper Scenario Change: Episode count: {self.workflow_episode} Seed :{self.seed}")
            self.step_index = 0
        else:
            # Continue with the same fault scenario..
            self.state = self.initial_state[self.seed]
            self.step_index = 0
            
        return self.state
