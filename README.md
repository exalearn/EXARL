# Reinforcement Learning environments and agents/policies used for the Design and Control applications

## Software Requirement
* Python 3.7 
* The ExaRL framework is built on [OpenAI Gym](https://gym.openai.com) 
* Additional python packages are defined in the setup.py 
* This document assumes you are running at the top directory 

## Directory Organization
```
├── setup.py                          : Python setup file with requirements files 
├── scripts                           : folder containing RL steering scripts
├── mpi_scripts                       : folder containing RL MPI steering scripts
├── exaRL                             : folder containing wrapper script for unified syntax
├── exa_agents                        : folder containing ExaRL agents and registration scripts
    └── __init__.py                   : agent registry
    ├── agents                        : folder containing agents
        └── __init__.py               : script to make agents visible
        ├── agent_cfg                 : folder containing default agent configurations   
├── exa_envs                          : folder containing ExaRL environments
    └── __init__.py                   : environment registry
    ├── envs                          : folder containing environments
    └── __init__.py                   : script to make environments visible
        ├── env_cfg                   : folder containing default environment configurations    
├── utils                             : folder containing utilities       
```

## Installing 
* Pull code from repo
```
git clone https://github.com/exalearn/ExaRL.git
```
* Install dependencies for ExaRL:
```
cd ExaRL
pip install -e . --user
```

## Running ExaRL using MPI 
```
mpiexec -np 3 python mpi_scripts/mpi_dqn_exacartpole.py
```

## Creating custom environments
* ExaRL uses OpenAI gym environments
* Register the environment in ```ExaRl/exa_envs/__init__.py```
    
```
from gym.envs.registration import register

register(
    id='fooEnv-v0',
    entry_point='exa_envs.envs:FooEnv',
)
```
* The id variable will pass to exaRL.make() to call environment

* The file ```ExaRL/exa_env/envs/__init__.py``` should include
```
from exa_envs.envs.foo_env import FooEnv
```
where ExaRL/exa_envs/envs/foo_env.py is the file containing your envirnoment

## Creating custom agents
* ExaRL extends OpenAI gym's environment registration to agents
* Register the agent in ```ExaRL/exa_agents/__init__.py```
    
```
from .registration import register, make

register(
    id='fooAgent-v0',
    entry_point='exa_agents.agents:FooAgent',
)
```
* The id variable will pass to exaRL.make() to call agent

* The file ```ExaRL/exa_agents/agents/__init__.py``` should include
```
from exa_agents.agents.foo_agent import FooAgent
```
where ExaRL/exa_agents/agents/foo_agent.py is the file containing your agent

## Calling agents and environments in your scripts
* ExaRL uses a unified syntax to call agents and environments
```
import exaRL as erl
agent, env = erl.make('fooAgent-v0', 'fooEnv-v0')
```


