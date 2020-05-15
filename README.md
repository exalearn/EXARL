![](EXARL.png)
# Easily eXtendable Architecture for Reinforcement Learning
A scalable software framework for reinforcement learning environments and agents/policies used for the Design and Control applications

## Software Requirement
* Python 3.7 
* The EXARL framework is built on [OpenAI Gym](https://gym.openai.com) 
* Additional python packages are defined in the setup.py 
* This document assumes you are running at the top directory 

## Directory Organization
```
├── setup.py                          : Python setup file with requirements files
├── learner_cfg.json                  : Learner configuration file
├── scripts                           : folder containing RL steering scripts
├── driver                            : folder containing RL MPI steering scripts
    └── driver_example.py             : Example run scipt
    └── candleDriver.py               : Supporting CANDLE script
├── candlelib                         : folder containing library for CANDLE functionality
├── exarl                	          : folder containing base classes
    └── __init__.py                   : make base classes visible
    └── agent_base.py                 : agent base class
    └── env_base.py                   : environment base class
    └── learner_base.py               : learner base class
├── agents         	                  : folder containing ExaRL agents and registration scripts
    └── __init__.py                   : agent registry
    ├── agent_vault                   : folder containing agents
        └── __init__.py               : script to make agents visible
        ├── agent_cfg                 : folder containing default agent configurations   
├── envs         	                  : folder containing ExaRL environments
    └── __init__.py                   : environment registry
    ├── env_vault                     : folder containing environments
    └── __init__.py                   : script to make environments visible
        ├── env_cfg                   : folder containing default environment configurations    
├── utils                             : folder containing utilities       
```

## Installing 
* Pull code from repo
```
git clone --recursive https://github.com/exalearn/ExaRL.git
```
* Install dependencies for ExaRL:
```
cd ExaRL
pip install -e . --user
```

## [CANDLE](https://github.com/ECP-CANDLE/Candle) functionality is built into EXARL
* Add/modify the learner parameters in ```ExaRL/learner_cfg.json```\
E.g.:-
```
{
    "agent": "DQN-v0",
    "env": "ExaLearnCartpole-v1",
    "n_episodes": 1,
    "n_steps": 10,
    "mpi_children_per_parent": 3,
    "run_type": "static",
    "output_dir": "./exa_results_dir"
}
```
* Add/modify the agent parameters in ```ExaRL/agents/agent_vault/agent_cfg/<AgentName>.json```\
E.g.:-
```
{
    "search_method": "epsilon",
    "gamma": 0.95,
    "epsilon": 1.0,
    "epsilon_min" : 0.1,
    "epsilon_decay" : 0.995,
    "learning_rate" : 0.01,
    "batch_size" : 10,
    "tau" : 1.0,
    "dense" : [64, 128],
    "activation" : "relu",
    "optimizer" : "adam",
    "loss" : "mse"
}
```
* Add/modify the learner parameters in ```ExaRL/envs/env_vault/env_cfg/<EnvName>.json```\
E.g.:-
```
{
        "mpi_children_per_parent": "3",
        "worker_app": "./envs/env_vault/cpi.py"
}
```
* Please note the agent and environment configuration file (json file) name must match the agent and environment ID specified in ```ExaRL/learner_cfg.json```. \
E.g.:- ```ExaRL/agents/agent_vault/agent_cfg/DQN-v0.json``` and ```ExaRL/envs/env_vault/env_cfg/ExaLearnCartpole-v1.json```

## Running EXARL using MPI
* Existing environment can be paired with an available agent
* The following script is provided for convenience: ```ExaRL/driver/driver_example.py```
```
import exarl as erl
import driver.candleDriver as cd

## Get run parameters using CANDLE
run_params = cd.initialize_parameters()

## Create learner object and run
exa_learner = erl.ExaLearner(run_params)
run_type = run_params['run_type'] # can be either 'static' or 'dynamic'
exa_learner.run(run_type)
```
* Write your own script or modify the above as needed
* Run the following command:
```
mpiexec -np <num_parent_processes> python driver/driver_example.py --<run_params>=<param_value>
```
* The ```get_config()``` method is available in the base classes ```ExaRL/exarl/agent_base.py``` and ```ExaRL/exarl/env_base.py``` to obtain the parameters from CANDLE.

### Using parameters set in CANDLE configuration/get parameters from terminal
* Declare the parameters in the constructor of your agent/environment class
* Initialize the parameters to have proper datatypes
For example: 
```
self.search_method = '' # string type
self.gamma = 0.0 # float type
```
* The parameters can be fetched from CANDLE as: ```config = super().get_config()```
* Individual parameters are accessed using the corresponding key
```
self.search_method =  (agent_data['search_method'])
self.gamma =  (agent_data['gamma'])

```

## Creating custom environments
* ExaRL uses OpenAI gym environments
* Environments inherit from gym.Env and exarl.ExaEnv
```
Example:-
    class envName(gym.Env, exarl.ExaEnv):
        ...
```
* Environments must include the following variables and functions:
```
# Use 'dynamic' for dynamic MPI process spawning, else 'static' 
self.run_type = <`static` or 'dynamic'>
step()      # returns new state after an action
reset()     # reset the environment to initial state; marks end of an episode
set_env()   # set environment hyperparameters
```
* Register the environment in ```ExaRl/envs/__init__.py```
    
```
from gym.envs.registration import register

register(
    id='fooEnv-v0',
    entry_point='envs.env_vault:FooEnv',
)
```
* The id variable will be passed to exarl.make() to call the environment

* The file ```ExaRL/env/env_vault/__init__.py``` should include
```
from envs.env_vault.foo_env import FooEnv
```
where ExaRL/envs/env_vault/foo_env.py is the file containing your envirnoment

### Using environment written in a lower level language
* The following example illustrates using the C function of computing the value of PI in EXARL \
computePI.h:
```
#define MPICH_SKIP_MPICXX 1
#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif
  extern void compute_pi(int, MPI_Comm);
#ifdef __cplusplus
}
#endif
```

computePI.c:
```
#include <stdio.h>
#include <mpi.h>

double compute_pi(int N, MPI_Comm new_comm)
{
  int rank, size;
  MPI_Comm_rank(new_comm, &rank);
  MPI_Comm_size(new_comm, &size);

  double h, s, x;
  h = 1.0 / (double) N;
  s = 0.0;
  for(int i=rank; i<N; i+=size)
  {
    x = h * ((double)i + 0.5);
    s += 4.0 / (1.0 + x*x);
  }
  return (s * h);
}
```
* Compile the C/C++ code and create a shared object (*.so file)
* Create a python wrapper (Ctypes wrapper is shown) \
\
computePI.py:
```
from mpi4py import MPI
import ctypes
import os

_libdir = os.path.dirname(__file__)

if MPI._sizeof(MPI.Comm) == ctypes.sizeof(ctypes.c_int):
    MPI_Comm = ctypes.c_int
else:
    MPI_Comm = ctypes.c_void_p
_lib = ctypes.CDLL(os.path.join(_libdir, "libcomputePI.so"))
_lib.compute_pi.restype = ctypes.c_double
_lib.compute_pi.argtypes = [ctypes.c_int, MPI_Comm]

def compute_pi(N, comm):
    comm_ptr = MPI._addressof(comm)
    comm_val = MPI_Comm.from_address(comm_ptr)
    myPI = _lib.compute_pi(ctypes.c_int(N), comm_val)
    return myPI
```
* In your environment code, just import the function and use it regularly \
test_computePI.py:
```
from mpi4py import MPI
import numpy as np
import pdb
import computePI as cp

def main():
    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()
    nprocs = comm.Get_size()

    if myrank == 0:
        N = 100
    else:
        N = None

    N = comm.bcast(N, root=0)
    num = 4
    color = int(myrank/num)
    newcomm = comm.Split(color, myrank)

    mypi = cp.compute_pi(N, newcomm)
    pi = newcomm.reduce(mypi, op=MPI.SUM, root=0)

    newrank = newcomm.rank
    if newrank==0:
        print(pi)

if __name__ == '__main__':
    main()

```

## Creating custom agents
* EXARL extends OpenAI gym's environment registration to agents
* Agents inherit from exarl.ExaAgent
```
Example:-
    class agentName(exarl.ExaAgent):
        ...
```
* Agents must include the following functions:
```
get_weights()   # get target model weights
set_weights()   # set target model weights
train()         # train the agent
update()        # update target model
action()        # Next action based on current state
load()          # load weights from memory
save()          # save weights to memory
monitor()       # monitor progress of learning
set_agent()     # set agent hyperparameters
```
* Register the agent in ```ExaRL/agents/__init__.py```
    
```
from .registration import register, make

register(
    id='fooAgent-v0',
    entry_point='agents.agent_vault:FooAgent',
)
```
* The id variable will be passed to exarl.make() to call the agent

* The file ```ExaRL/agents/agent_vault/__init__.py``` should include
```
from agents.agent_vault.foo_agent import FooAgent
```
where ExaRL/agents/agent_vault/foo_agent.py is the file containing your agent

## Base classes
* Base classes are provided for agents, environments, and learner in the directory ```ExaRL/exarl/```
* Users can inherit from the correspoding agent and environment base classes

## Contacts
If you have any questions or concerns regarding EXARL, please contact Vinay Ramakrishnaiah (vinayr@lanl.gov) and/or Malachi Schram (Malachi.Schram@pnnl.gov).

