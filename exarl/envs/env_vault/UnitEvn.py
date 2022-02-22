import gym
from gym import spaces
import numpy as np
from exarl.base.comm_base import ExaComm

class EnvGenerator:
    """"
    This class is used to generate environments to use for testing.
    
    Attributes
    ----------
    high : np.array
        Used to set the upper and lower bound of gym spaces
    spaceDict : dictionary
        Contains a map of string to gym spaces to test
    """
    high = np.array([1, 1, 1], dtype=np.float64)
    spaceDict = {
        "Discrete": spaces.Discrete(5),
        "Box_One": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float64),
        "Box_Two": spaces.Box(low=-1, high=1, shape=(1, 2), dtype=np.float64),
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

    @staticmethod
    def createClass(action_space, observation_space, 
                    action_tuple, observation_tuple,
                    reset_flag, max_steps,
                    num_seeds):
        """
        This is the factory for new classes.

        Attributes
        ----------
        action_space : string
            String that maps to gym space for action
        observation_space : string
            String that maps to gym space for observation
        action_tuple : string
            Indicates if to use a tuple for action
        observation_tuple : string
            Indicates if to use a tuple for observation
        reset_flag : bool
            Indicates if reset should actually reset env
        max_steps : int
            The max steps before sending done after step
        num_seeds : int
            Number of unique seeds
        
        Returns
        -------
        gym.Env
            Environment to use for testing
        """
        class envTester(gym.Env):
            name = "_".join((action_space, observation_space, 
                            str(action_tuple), str(observation_tuple),
                            str(reset_flag), str(max_steps),
                            str(num_seeds))) + "-v0"

            def __init__(self):
                super().__init__()
                actSpace = EnvGenerator.spaceDict[action_space]
                obvSpace = EnvGenerator.spaceDict[observation_space]
                self.action_space = spaces.Tuple((actSpace, actSpace)) if action_tuple else actSpace
                self.observation_space = spaces.Tuple((obvSpace, obvSpace)) if observation_tuple else obvSpace
                self.num_seeds = num_seeds
                self.seed = 0
                self.initial_state = []
                for i in range(self.num_seeds):
                    self.initial_state.append(self.observation_space.sample())
                if ExaComm.is_actor():
                    self.initial_state = ExaComm.env_comm.bcast(self.initial_state, root=0)

                self.state = self.initial_state[self.seed]
                self.seed = (self.seed + 1) % self.num_seeds
                self.step_index = 0
                self.max_steps = max_steps
                
            def step(self, action):
                if self.step_index < self.max_steps:
                    self.state = self.observation_space.sample()
                    self.step_index += 1
                done = self.step_index == self.max_steps
                return self.state, 1, done, {}
                
            def reset(self):
                if reset_flag:
                    self.state = self.initial_state[self.seed]
                    self.seed = (self.seed + 1) % self.num_seeds
                return self.state
        
        return envTester

    @staticmethod
    def generator(reset_flag=True, max_steps=100, num_seeds=20):
        """
        This is the generator to iterate through different options.

        Attributes
        ----------
        reset_flag : True
            Indicates if classes generaged actually reset
        max_steps : int
            Max step of the classes generated
        num_seeds : int
            Number of seed of the classes generated
        
        Returns
        -------
        gym.env
            A tester environment
        """
        for act_tuple in [False, True]:
            for obs_tuple in [False, True]:
                for act_space in EnvGenerator.spaceDict:
                    for obs_space in EnvGenerator.spaceDict:
                        yield EnvGenerator.createClass(act_space, obs_space, act_tuple, obs_tuple, reset_flag, max_steps, num_seeds)

    @staticmethod
    def getNames():
        """
        Returns the names of the classes generated
        Returns
        -------
        List
            Names of the classes generated
        """
        return [entry.name for entry in EnvGenerator.generator()]
