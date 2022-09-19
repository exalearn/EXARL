from os import path
import os
import numpy as np
import gym
import gym.spaces as spaces
from typing import Any, Dict, Optional, Tuple, Union, Sequence
from exarl.base.comm_base import ExaComm
from exarl.utils.globals import ExaGlobals

import bsuite
from bsuite.utils import gym_wrapper

from dm_control import mujoco
from dm_control import mjcf
from dm_control import suite

from dm_env import specs

from PIL import Image
from PIL import ImageDraw


_GymTimestep = Tuple[np.ndarray, float, bool, Dict[str, Any]]




class ExaDMControl(gym.Env):
    def __init__(self) -> None:
        super().__init__()
        self.env_comm = ExaComm.env_comm
        self.episode = -1
        self.step_count = 0
        self.discrete = ExaGlobals.lookup_params("discrete")
        self.discrete_step = ExaGlobals.lookup_params("discrete_step")
        self.domain = ExaGlobals.lookup_params("domain")
        self.task = ExaGlobals.lookup_params("task")
        self.render = ExaGlobals.lookup_params("render") # Toggle render with boolean
        self.camera_id = ExaGlobals.lookup_params("camera_id") # 0 or 1 for camera positions
        

        # Let self.raw_env be of class dm_env.Environment.
        # Then return gym-like outputs for step, reset methods.
        self.raw_env = suite.load(self.domain, self.task, environment_kwargs={"flat_observation": True})
        self.env = gym_wrapper.GymFromDMEnv(self.raw_env)
        self.action_space, self.action_spec = self.build_action_space(self.raw_env)
        self.observation_space = self.build_observation_space(self.raw_env)
    
    def step(self, action) -> _GymTimestep:
        if self.discrete:
            timestep = self.raw_env.step(self.map(action))
        else:
            timestep = self.raw_env.step(action)
        if self.render:
            if self.step_count % 30 == 0:
                self.render_step()
        self.step_count+=1
        next_state = timestep.observation["observations"]
        reward = timestep.reward
        done = timestep.step_type.last()
        return next_state, reward, done, {}

    def reset(self) -> np.ndarray:
        timestep = self.raw_env.reset()  
        #print("episode: ", self.episode, flush=True)
        self.episode+=1
        self.step_count = 0 
        return timestep.observation["observations"]
    
    def render_step(self):
        pixels = self.raw_env.physics.render(camera_id=self.camera_id)
        im = Image.fromarray(pixels)
        img_d = ImageDraw.Draw(im)
        img_d.text((28, 36), str(self.episode), fill=(255, 0, 0))
        im.save("tmp/"+str(self.episode)+"-"+str(self.raw_env.physics.data.time)+".png")

    def build_action_space(self, dm_env):
        action_spec = dm_env.action_spec() # type: specs.DiscreteArray
        if self.discrete:
            return spaces.Discrete(               
                int((action_spec.maximum[0]-action_spec.minimum[0])/self.discrete_step)
            ), action_spec
            #return spaces.MultiDiscrete(int((action_spec.maximum-action_spec.minimum)/self.discrete_step)), action_spec
        print(action_spec.minimum)
        return spaces.Box(
            low=-1,
            high=1,
            shape=action_spec.shape,
            dtype=action_spec.dtype), action_spec

    def build_observation_space(self, dm_env):
        obs_spec = dm_env.observation_spec()  # type: specs.Array
        obs_spec = obs_spec["observations"]
        return spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=obs_spec.shape,
            dtype=obs_spec.dtype)
    def map(self, action):
        return (action*self.discrete_step + self.action_spec.minimum[0])
