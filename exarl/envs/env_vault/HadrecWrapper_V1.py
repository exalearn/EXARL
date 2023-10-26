from asyncio.log import logger
import gym
import time
import numpy as np
import sys
import json
import exarl as erl
from exarl.base.comm_base import ExaComm

import os
import gridpack
import gridpack.hadrec
import random
from gym.utils import seeding
from gym import spaces
import math
import xmltodict
import collections
import xml.etree.ElementTree as ET

from exarl.envs.env_vault.Hadrec_dir.exarl_env.Hadrec import Hadrec
from exarl.utils.globals import ExaGlobals

from exarl.base.comm_base import ExaComm

class HadrecWrapper_V1(gym.Env):

    def __init__(self):
        super().__init__()
        # rl_config_file,rl_config_file,simu_input_Rawfile,simu_input_Dyrfile
        self.agent_comm = ExaComm.agent_comm
        self.learner_comm = ExaComm.learner_comm
        # MPI_comm = ExaComm.get_MPI().COMM_WORLD
        # if ExaComm.env_comm is None:
        #     MPI_comm = (ExaComm.get_MPI().COMM_SELF) # This used for AMILES paper/ 
        # else:
        #     MPI_comm = ExaComm.env_comm.raw()

        MPI_comm = (ExaComm.get_MPI().COMM_SELF) 
        # MPI_comm = (ExaComm.get_MPI().COMM_WORLD)
        
        # MPI_comm = (ExaComm.get_MPI().COMM_SELF) # This used for AMILES paper/
        # print(MPI_comm)

        self.rl_config_file = ExaGlobals.lookup_params('rl_config_file')
        self.simu_input_file = ExaGlobals.lookup_params('simu_input_file')
        self.simu_input_Rawfile = ExaGlobals.lookup_params('simu_input_Rawfile')
        self.simu_input_Dyrfile = ExaGlobals.lookup_params('simu_input_Dyrfile')
        # This updates the input xml file with the required file location.
        # self.UpdateXMLFile()

        # print(self.simu_input_file,self.rl_config_file )
        self.env = Hadrec(simu_input_file=self.simu_input_file,
                          rl_config_file=self.rl_config_file,
                          Comm=MPI_comm)
        
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        self.workflow_episode = 0
        self.Num_perturb = 2

    def UpdateXMLFile(self):
        tree = ET.parse(self.simu_input_file)

        logger.info("Updating the XML file with the candle passed input data path")

        logger.info(tree.find("Powerflow/networkFiles/networkFile/networkConfiguration_v33").text)
        logger.info(tree.find("Dynamic_simulation/generatorParameters").text)

        (tree.find("Powerflow/networkFiles/networkFile/networkConfiguration_v33").text) = self.simu_input_Rawfile
        (tree.find("Dynamic_simulation/generatorParameters").text) = self.simu_input_Dyrfile

        tree.write(self.simu_input_file)

        return

    def step(self, action):
        return self.env.step(action)


    def reset(self):
        # if self.workflow_episode % self.Num_perturb == 0 and self.workflow_episode >= self.Num_perturb:
        #     # Reset the environment with the new fault case
        #     print(f"Wrapper Scenario Change resetting the gridpack environment Episode count: {self.workflow_episode} ")
        #     return self.env.reset(flag=1)
        # else:
        return self.env.reset(flag=1)

    # def reset(self):
    #     return self.env.reset()

    def set_env(self):
        return self.env.set_env()

    def seed(self, seed=None):
        return self.env.seed(seed)

        # ---------------initialize the system with a specific state and fault
    def validate(self, case_Idx, fault_bus_idx, fault_start_time, fault_duration_time):
        return self.env.validate(case_Idx, fault_bus_idx, fault_start_time, fault_duration_time)

    def close_env(self):
        return self.env.close_env()

    def get_base_cases(self):
        return self.env.get_bases_cases()
    
    def get_printCaseDetails(self):
        self.env.get_printCaseDetails()

