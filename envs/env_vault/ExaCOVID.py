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
           - Increase 
           - 

        Reward is the number of people infected (??)

        """
        ## /home/schr476/exalearn/corvid/corviddata
        self.cfg_data = super.get_config()
        self.app = ''
        self.workder_dir = ''

        ## Define state and action spaces
        self.policies_len  =1 ## Only using workfromhome policy for now
        self.workfraction_step =0.1
        self.observation_space = spaces.Box(low=np.append(np.zeros(self.policies_len)), high=np.append(np.ones(self.policies_len)),dtype=np.float32)

        ## Increase, Decrease, Don't change
        self.action_space = spaces.Discrete(3) 

        ## Create model parameter ##
        self.model_parameter = {}

    def step(self, action):
        ## Initial step variable ##
        done   = False
        reward = 0
        info   = ''
        
        ## Apply discrete actions
        if action==1:
            self.model_parameter['workfromhome']+=self.workfraction_step 
            if self.model_parameter['workfromhome']>self.observation_space.high[0]:
                self.model_parameter['workfromhome']-=self.workfraction_step 
                done = True
                return self._getState(),reward,done, {}
                
        elif action==2:
            self.model_parameter['workfromhome']-=self.workfraction_step 
            if self.model_parameter['workfromhome']<self.observation_space.high[0]:
                self.model_parameter['workfromhome']+=self.workfraction_step 
                done = True
                return self._getState(),reward,done, {}

        ## Make input file ##
            
        ## Run model state ##
        env_out = subprocess.Popen([self.app], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=self.worker_dir)
        stdout,stderr = env_out.communicate()
        logger.info(stdout)
        logger.info(stderr)
        
        ## Calculate the number of infected ##
        reward = self.__get_healthy_sum(filename)
 
        
        return next_state, reward, done, info

    def __make_input_file(self,filename):
        ''' Description:
              Creates a new input file based on the action
        '''
        return 0
    
    def __get_healthy_sum(self,filename):
        ''' Description:
             Calculates the total number of symptomatic people 
        '''
        total_people  = 0
        total_healthy = 0
        total_symp    = 0
        f = open(filename, "r")
        for x in f:
            if 'People:' in x:
                total_people = int(x.split()[-1])
            if 'Total symptomatic individuals by age:' in x:
                total_symp = sum( int(x) for x in ((x.split())[-1].split(','))[0:-1])

        total_healthy = total_people - total_symp 
        return total_healthy
    
    def reset(self):

    def render(self):

