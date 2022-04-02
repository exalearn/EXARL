import os
import gym
import numpy as np
import exarl as erl
from exarl.base.comm_base import ExaComm
import exarl.utils.candleDriver as cd

class ScriptEnv(gym.Env):

    def __init__(self):
        super().__init__()
        # Create spaces
        float_max = np.finfo(np.float64).max
        low = np.array([0, 0], dtype=np.float64)
        high = np.array([float_max, float_max], dtype=np.float64)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float64)
        self.action_space = gym.spaces.Discrete(1)
        
        self.command = (
            'mpirun -n 1 ' +
            '/people/firo017/sst_macro/sst-macro/build/install/bin/sstmac ' + 
            '-f /people/firo017/sst_macro/sst-macro/build/random/parameters_offered_load_random_lps_8k_2p00ms.ini ' +
            '-p topology.name=file ' + 
            '-p topology.filename=/people/firo017/sst_macro/sst-macro/build/test_graph.json ' +
            '-p topology.routing_tables=/people/firo017/sst_macro/sst-macro/build/test_graph_routes.json' 
            )
        
        # Init states
        self.initial_state = low
        self.state = low

    def getCommand(self, filename, r):
        command = (
            'mpirun -n 1 ' +
            '/people/firo017/sst_macro/sst-macro/build/install/bin/sstmac ' + 
            '-f /people/firo017/sst_macro/sst-macro/build/random/parameters_offered_load_random_lps_8k_2p00ms.ini ' +
            '-p topology.name=file ' + 
            '-p topology.filename=/people/firo017/sst_macro/sst-macro/build/test_graph.json ' +
            '-p topology.routing_tables=/people/firo017/sst_macro/sst-macro/build/test_graph_routes.json' 
            )

    def step(self, action):
        # Run Command
        stream = os.popen(self.command)
        output = stream.read()
        lines = output.split("\n")
        
        # Process results
        runtime_line = [line for line in lines if "Estimated total runtime of" in line]
        runtime = [np.float64(x) for x in runtime_line[0].split() if "." in x]
        exp_time_line = [line for line in lines if "ST/macro ran for" in line]
        exp_time = [np.float64(x) for x in exp_time_line[0].split() if "." in x]

        # Generate the next state
        next_state = np.array([runtime[0], exp_time[0]], dtype=np.float64)
        print(next_state)
        return next_state, 1, True, {}

    def reset(self):
        self.state = self.initial_state
        return self.state
