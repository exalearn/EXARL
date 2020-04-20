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
├── scripts                           : folder containing RL steering scripts
├── driver                            : folder containing RL MPI steering scripts
    └── exalearn_example.py           : Example run scipt
    └── candle_example.py             : Example run script that includes CANDLE functionality
    └── candleDriver.py               : Supporting CANDLE script
├── candlelib                         : folder containing library for CANDLE functionality
├── exarl                	          : folder containing base classes
    └── __init__.py                   : make base classes visible
    └── wrapper.py                    : wrapper for unified syntax
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
git clone https://github.com/exalearn/ExaRL.git
```
* Install dependencies for ExaRL:
```
cd ExaRL
pip install -e . --user
```

## Running ExaRL using MPI
* Existing environment can be paired with an available agent
* The following script is provided for convenience: ```ExaRL/driver/exalearn_example.py```
```
import exarl as erl

## Define agent and env
agent_id = 'agents:DQN-v0' # Specify agent
env_id   = 'envs:ExaLearnCartpole-v1' # Specify env

## Create learner
exa_learner = erl.ExaLearner(agent_id,env_id)
exa_learner.set_results_dir('/gpfs/alpine/ast153/scratch/vinayr/')
exa_learner.set_training(10,10) # (num_episodes, num_steps)
run_type = exa_learner.env.run_type
exa_learner.run(run_type)
```
* Write your own script or modify the above as needed
* Run the following command:
```
mpiexec -np <num_parent_processes> python driver/exalearn_example.py
```
## Using CANDLE functionality
* Default parameters are in ```ExaRL/combo_setup.txt```
```
[Driver Params]
output_dir = './exa_results_dir'
agent = 'DQN-v0'
env = 'ExaLearnCartpole-v1'

[Learner Params]
n_episodes = 1
n_steps = 10

[DQN_Params]
search_method = 'epsilon'
gamma = .95
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
learning_rate = 0.01
batch_size = 10
tau = 1.0
```
* Refer to the script ```ExaRL/driver/candle_example.py```
```
import exarl as erl
import driver.candleDriver as cd

## Get run parameters using CANDLE
run_params = cd.initialize_parameters()
results_dir = run_params['output_dir']+run_params['experiment_id']+'/'+run_params['run_id']

## Define agent and env
agent_id = 'agents:'+run_params['agent']
env_id   = 'envs:'+run_params['env']

## Create learner object and run
exa_learner = erl.ExaLearner(agent_id,env_id)
exa_learner.set_config(run_params)
exa_learner.set_results_dir(results_dir)
run_type = exa_learner.env.run_type
exa_learner.run(run_type)
```
* Write your own script or modify the above as needed
* Run the following command:
```
mpiexec -np <num_parent_processes> python driver/exalearn_example.py --<run_params>=<param_value>
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
render()    # render the environment
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

## Creating custom agents
* ExaRL extends OpenAI gym's environment registration to agents
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

## Calling agents and environments in custom learner scripts
* ExaRL uses a unified syntax to call agents and environments
```
import exarl as erl
agent, env = erl.make('fooAgent-v0', 'fooEnv-v0')
```
* This functionality is demonstrated in ```ExaRL/exarl/learner_base.py```

## Best practices
* Include a .json file in ```ExaRL/envs/env_vault/env_cfg/``` for environment related configurations
* Include a .json file in ```ExaRL/agents/agent_vault/agent_cfg/``` for agent related configurations

## Configurations
* If configurations are not provided, the following defaults will be used:

* Agent defaults:
```
default_agent_cfg = agents/agent_vault/agent_cfg/dqn_setup.json
search_method     = epsilon
gamma             = 0.95
epsilon           = 1.0
epsilon_min       = 0.05
epsilon_decay     = 0.995
learning_rate     = 0.001
batch_size        = 32    # memory batch size for training
tau               = 0.5
```

* Environment defaults:
```
default_env_cfg         = envs/env_vault/env_cfg/env_setup.json
mpi_children_per_parent  = 0
worker                  = envs/env_vault/cpi.py  # Synthetic workload that computes PI (runs only if mpi_children_per_parent > 0)
```

## Base classes
* Base classes are provided for agents, environments, and learner
* The learner base class (ExaLearner) includes the following functions:
```
set_training()      # set number of episodes and steps per episode
set_config()        # set hyperparameters using CANDLE
set_results_dir()   # result directory path
render_env()        # True or False
run_exarl()         # run learner
run()               # Setup to run static or dynamic learner
```

## Contacts
If you have any questions or concerns regarding EXARL, please contact Vinay Ramakrishnaiah (vinayr@lanl.gov) and/or Malachi Schram (Malachi.Schram@pnnl.gov).

