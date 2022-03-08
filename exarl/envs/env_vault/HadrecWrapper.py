import gym
import time
import numpy as np
import sys
import json
import exarl as erl
from exarl.base.comm_base import ExaComm

import sys, os
import gridpack
import gridpack.hadrec
import random
from gym.utils import seeding
from gym import spaces
import math
import xmltodict
import collections

from exarl.envs.env_vault.Hadrec_dir.exarl_env.Hadrec import Hadrec

class HadrecWrapper(gym.Env):


    def __init__(self):
        super().__init__()
        self.env = Hadrec(simu_input_file="/global/homes/t/tflynn/powerGridEnv/testData/IEEE39/input_39bus_step005_training_v33_newacloadperc43_multipf.xml",
                          rl_config_file="/global/homes/t/tflynn/powerGridEnv/testData/IEEE39/json/IEEE39_RL_loadShedding_3motor_5ft_gp_lstm.json",
)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
    def step(self, action):
        return self.env.step(action)
    
    def reset(self):
        return self.env.reset()
    
    def set_env(self):
        return self.env.set_env()

    def seed(self, seed=None):
        return self.env.seed(seed)

        #---------------initialize the system with a specific state and fault
    def validate(self, case_Idx, fault_bus_idx, fault_start_time, fault_duration_time):
        return self.env.validate(case_Idx, fault_bus_idx, fault_start_time, fault_duration_time)
    
    def close_env(self):
        return self.env.close_env()
    
    def get_base_cases(self):
        return self.env.get_bases_cases()
