![](EXARL.png)

# Easily eXtendable Architecture for Reinforcement Learning

A scalable software framework for reinforcement learning environments and agents/policies used for the Design and Control applications

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Build Status](https://travis-ci.com/exalearn/EXARL.svg?token=nVtzNrBfRo4qpVpEQP21&branch=develop)](https://travis-ci.com/exalearn/EXARL)
[![Documentation Status](https://readthedocs.org/projects/exarl/badge/?version=latest)](https://exarl.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/exalearn/EXARL/branch/develop/graph/badge.svg?token=VMHFSSZ7MJ)](https://codecov.io/gh/exalearn/EXARL)

[Complete documentation](https://exarl.readthedocs.io/en/latest/index.html) is available.

## Software Requirement

- Python 3.7
- The EXARL framework is built on [OpenAI Gym](https://gym.openai.com)
- Additional python packages are defined in the setup.py
- This document assumes you are running at the top directory

## Directory Organization

```
├── setup.py                          : Python setup file with requirements files
├── scripts                           : folder containing RL steering scripts
├── config                	          : folder containing configurations
    └── agent_cfg                     : agent configuration folder
    └── model_cfg                     : model configuration folder
    └── env_cfg                       : env configuration folder
    └── workflow_cfg                  : workflow configuration folder
    └── learner_cfg.json              : learner configuration
├── exarl                	            : folder with EXARL code
    └── __init__.py                   : make base classes visible
    └── mpi_settings.py               : MPI settings
    ├── base         	                : folder containing EXARL base classes
        └── __init__.py               : make base classes visible
        └── agent_base.py             : agent base class
        └── env_base.py               : environment base class
        └── workflow_base.py          : workflow base class
        └── learner_base.py           : learner base class
    ├── driver                        : folder containing RL MPI steering scripts
        └── driver.py                 : Run scipt
    ├── candlelib                     : folder containing library for CANDLE functionality
    ├── agents         	              : folder containing EXARL agents and registration scripts
        └── __init__.py               : agent registry
        └── registration.py           : script to handle registration
        ├── agent_vault               : folder containing agents
            └── __init__.py           : script to make agents visible
            └── <RLagent>.py          : RL agents (such as DQN, DDPG, etc.)
    ├── envs         	                : folder containing EXARL environments
        └── __init__.py               : environment registry
        ├── env_vault                 : folder containing environments
        └── __init__.py               : script to make environments visible
            └── <RLenv>.py            : RL environments (physics simulations, interfaces to experiments, etc.)
    ├── workflows      	              : folder containing EXARL workflows and registration scripts
        └── __init__.py               : workflow registry
        └── registration.py           : script to handle registration
        ├── workflow_vault            : folder containing workflows
            └── __init__.py           : script to make workflows visible
            └── <RLworkflow>.py       : RL workflows (such as SEED, IMPALA, etc.)
    ├── utils                         : folder containing utilities
        └── __init__.py               : make classes and functions visible
        └── candleDriver.py           : Supporting CANDLE script
        └── analyze_reward.py         : script for plotting results
        └── log.py                    : central place to set logging levels
        └── profile.py                : provides function decorators for profiling, timing, and debugging
```

## Installing

- Clone the repository
- Note: This repository uses git large file system (lfs) for storing data. Make sure your git version supports lfs before cloning the repository.

```
git clone --recursive https://github.com/exalearn/EXARL.git
cd EXARL
# Required for older versions of git
git lfs install # install git lfs if you haven't
git lfs fetch
git lfs pull
```

- EXARL uses MPI and GPU accelerated versions of TF/PyTorch/Keras when available.
- Install dependencies for EXARL (Refer GitHub wiki for platform specific build instructions):

```
pip install -e setup/ --user
```

## Configuration Files

Configuration files such as `exarl/config/learner_cfg.json` are searched for in the
following directories:

1. (current working directory)/exarl/config
2. ~/.exarl/config
3. (site-packages dir)/exarl/config

If you would like to run EXARL from outside the source directory, you may
install the config files with exarl or copy them into EXARL's config directory
in your home directory like so:

```console
$ mkdir -p ~/.exarl/config
$ cd EXARL
$ cp config/* ~/.exarl/config
```

## [CANDLE](https://github.com/ECP-CANDLE/Candle) functionality is built into EXARL

- Add/modify the learner parameters in `EXARL/exarl/config/learner_cfg.json`\
  E.g.:-

```
{
    "agent": "DQN-v0",
    "env": "ExaLearnCartpole-v1",
    "workflow": "async",
    "n_episodes": 1,
    "n_steps": 10,
    "output_dir": "./exa_results_dir"
}
```

- Add/modify the agent parameters in `EXARL/exarl/config/agent_cfg/<AgentName>.json`\
  E.g.:-

```
{
    "gamma": 0.75,
    "epsilon": 1.0,
    "epsilon_min" : 0.01,
    "epsilon_decay" : 0.999,
    "learning_rate" : 0.001,
    "batch_size" : 5,
    "tau" : 0.5
}
```

Currently, DQN agent takes either MLP or LSTM as model_type.

- Add/modify the model parameters in `EXARL/exarl/config/model_cfg/<ModelName>.json`\
  E.g.:-

```
{
    "dense" : [64, 128],
    "activation" : "relu",
    "optimizer" : "adam",
    "out_activation" : "linear",
    "loss" : "mse"
}
```

- Add/modify the environment parameters in `EXARL/exarl/config/env_cfg/<EnvName>.json`\
  E.g.:-

```
{
        "worker_app": "./exarl/envs/env_vault/cpi.py"
}
```

- Add/modify the workflow parameters in `EXARL/exarl/config/workflow_cfg/<WorkflowName>.json`\
  E.g.:-

```
{
        "process_per_env": "1"
}
```

- Please note the agent, model, environment, and workflow configuration file (json file) name must match the agent, model, environment, and workflow ID specified in `EXARL/exarl/config/learner_cfg.json`. \
  E.g.:- `EXARL/exarl/config/agent_cfg/DQN-v0.json`, `EXARL/exarl/config/model_cfg/MLP.json`, `EXARL/exarl/config/env_cfg/ExaCartPole-v1.json`, and `EXARL/exarl/config/workflow_cfg/async.json`

## Running EXARL using MPI

- Existing environment can be paired with an available agent
- The following script is provided for convenience: `EXARL/exarl/driver/__main__.py`

```
from mpi4py import MPI
import utils.analyze_reward as ar
import time
import exarl as erl
import mpi4py.rc
mpi4py.rc.threads = False
mpi4py.rc.recv_mprobe = False

# MPI communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Get run parameters using CANDLE
# run_params = initialize_parameters()

# Create learner object and run
exa_learner = erl.ExaLearner(comm)

# Run the learner, measure time
start = time.time()
exa_learner.run()
elapse = time.time() - start

# Compute and print average time
max_elapse = comm.reduce(elapse, op=MPI.MAX, root=0)
elapse = comm.reduce(elapse, op=MPI.SUM, root=0)

if rank == 0:
    print("Average elapsed time = ", elapse / size)
    print("Maximum elapsed time = ", max_elapse)
    # Save rewards vs. episodes plot
    ar.save_reward_plot()
```

- Write your own script or modify the above as needed
- Run the following command:

```
mpiexec -np <num_parent_processes> python exarl/driver/__main__.py --<run_params> <param_value>
```

- If running a multi-process environment or agent, the communicators are available in `exarl/mpi_settings.py`.
  E.g.:-

```
import exarl.mpi_settings as mpi_settings
self.env_comm = mpi_settings.env_comm
self.agent_comm = mpi_settings.agent_comm
```

### Using parameters set in CANDLE configuration/get parameters from terminal

- To obtain the parameters from JSON file/set in terminal using CANDLE, use the following lines:

```
import exarl.utils.candleDriver as cd
cd.run_params # dictionary containing all parameters
```

- Individual parameters are accessed using the corresponding key \
  E.g.-

```
self.search_method =  cd.run_params['search_method']
self.gamma =  cd.run_params['gamma']

```

## Creating custom environments

- EXARL uses OpenAI gym environments
- The ExaEnv class in `EXARL/exarl/env_base.py` inherits from OpenAI GYM Wrapper class for including added functionality.
- Environments inherit from gym.Env

```
Example:-
    class envName(gym.Env):
        ...
```

- Register the environment in `EXARL/exarl/envs/__init__.py`

```
from gym.envs.registration import register

register(
    id='fooEnv-v0',
    entry_point='envs.env_vault:FooEnv',
)
```

- The id variable will be passed to exarl.make() to call the environment
- The file `EXARL/exarl/env/env_vault/__init__.py` should include

```
from exarl.envs.env_vault.foo_env import FooEnv
```

where EXARL/exarl/envs/env_vault/foo_env.py is the file containing your envirnoment

### Using environment written in a lower level language

- The following example illustrates using the C function of computing the value of PI in EXARL \
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

- Compile the C/C++ code and create a shared object (\*.so file)
- Create a python wrapper (Ctypes wrapper is shown) \
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

- In your environment code, just import the function and use it regularly \
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

- EXARL extends OpenAI gym's environment registration to agents
- Agents inherit from exarl.ExaAgent

```
Example:-
    import exarl
    class agentName(exarl.ExaAgent):
          def __init__(self, env, is_learner):
              ...
```

- Agents must include the following functions:

```
get_weights()   # get target model weights
set_weights()   # set target model weights
train()         # train the agent
update()        # update target model
action()        # Next action based on current state
load()          # load weights from memory
save()          # save weights to memory
monitor()       # monitor progress of learning
```

- Register the agent in `EXARL/exarl/agents/__init__.py`

```
from exarl.agents.registration import register, make

register(
    id='fooAgent-v0',
    entry_point='exarl.agents.agent_vault:FooAgent',
)
```

- The id variable will be passed to exarl.make() to call the agent
- The file `EXARL/exarl/agents/agent_vault/__init__.py` should include

```
from exarl.agents.agent_vault.foo_agent import FooAgent
```

where EXARL/agents/agent_vault/foo_agent.py is the file containing your agent

## Creating custom workflows

- EXARL also extends OpenAI gym's environment registration to workflows
- Workflows inherit from exarl.ExaWorkflow

```
Example:-
    class workflowName(exarl.ExaWorkflow):
        ...
```

- Workflows must include the following functions:

```
run()   # run the workflow
```

- Register the workflow in `EXARL/exarl/workflows/__init__.py`

```
from exarl.agents.registration import register, make

register(
    id='fooWorkflow-v0',
    entry_point='exarl.workflows.workflow_vault:FooWorkflow',
)
```

- The id variable will be passed to exarl.make() to call the agent
- The file `EXARL/exarl/workflows/workflow_vault/__init__.py` should include

```
from exarl.workflows.workflow_vault.foo_workflow import FooWorkflow
```

where EXARL/workflows/workflow_vault/foo_workflow.py is the file containing your workflow

## Base classes

- Base classes are provided for agents, environments, workflows, and learner in the directory `EXARL/exarl/`
- Users can inherit from the correspoding agent, environment, and workflow base classes

## Debugging, Timing, and Profiling

- Function decorators are provided for debugging, timing, and profiling EXARL.
- Debugger captures the function signature and return values.
- Timer prints execution time in seconds.
- Either line_profiler or memory_profiler can be used for profiling the code.
  - Profiler can be selected in `learner_cfg.json` or using the command line argument `--profile`.
  - Options for profiling are `line`, `mem`, or `none`.
- Function decorators can be used as shown below:

```
from exarl.utils.profile import *

@DEBUG
def my_func(*args, **kwargs):
    ...

@TIMER
def my_func(*args, **kwargs):
    ...

@PROFILE
def my_func(*args, **kwargs):
    ...
```

- Profiling results are written to: `results_dir + '/Profile/<line/memory>_profile.txt`.

## Cite this software

```
@misc{EXARL,
  author = {Vinay Ramakrishnaiah, Malachi Schram, Joshua Suetterlein, Jamal Mohd-Yusof, Sayan Ghosh, Yunzhi Huang, Ai Kagawa, Christine Sweeney, Shinjae Yoo},
  title = {Easily eXtendable Architecture for Reinforcement Learning (EXARL)},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/exalearn/EXARL}},
}
```

## Contacts

If you have any questions or concerns regarding EXARL, please contact Vinay Ramakrishnaiah (vinayr@lanl.gov), Joshua Suetterlein (joshua.suetterlein@pnnl.gov) or Jamal Mohd-Yusof (jamal@lanl.gov).
