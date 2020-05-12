import gym
import time
import numpy as np
import sys
import json
import exarl as erl

class ExaCOVID(gym.Env, erl.ExaEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, cfg='envs/env_vault/env_cfg/covid_env_setup.json'):
        """ 
        Description:
           Environment used to run the CORVID code

        Model source code:
           https://github.com/balewski/exaLearnEpi/tree/master/corvid_march

        Observation states map directly to the model input parameters that are being changed by the actions
           - 
           - 

        Action space is discrete: 
           - 
           - 

        Reward is the number of people infected (??)

        """
        ## /home/schr476/exalearn/corvid/corviddata
        
        self.app = ''
        self.workder_dir = ''

    def step(self, action):
        ## Initial step variable ##
        done   = False
        reward = 0
        info   = ''
        ## Run model state ##
        env_out = subprocess.Popen([self.app], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=self.worker_dir)
        stdout,stderr = env_out.communicate()
        
        logger.info(stdout)
        logger.info(stderr)
        
        return next_state, reward, done, info

    def reset(self):

    def render(self):

