Directory Organization
**********************

| ├── setup.py                          : Python setup file with requirements files
| ├── learner_cfg.json                  : Learner configuration file
| ├── scripts                           : folder containing RL steering scripts
| ├── driver                            : folder containing RL MPI steering scripts
|     └── driver.py                     : Run scipt 
| ├── candlelib                         : folder containing library for CANDLE functionality
| ├── exarl                	            : folder containing base classes
|     └── __init__.py                   : make base classes visible
|     └── agent_base.py                 : agent base class
|     └── env_base.py                   : environment base class
|     └── workflow_base.py              : workflow base class
|     └── learner_base.py               : learner base class
| ├── agents         	                : folder containing ExaRL agents and registration scripts
|     └── __init__.py                   : agent registry
|     └── registration.py               : script to handle registration
|     ├── agent_vault                   : folder containing agents
|         └── __init__.py               : script to make agents visible
|         ├── agent_cfg                 : folder containing default agent configurations
|         └── <RLagent>.py              : RL agents (such as DQN, DDPG, etc.)
| ├── envs         	                    : folder containing ExaRL environments
|     └── __init__.py                   : environment registry
|     ├── env_vault                     : folder containing environments
|     └── __init__.py                   : script to make environments visible
|         ├── env_cfg                   : folder containing default environment configurations
|         └── <RLenv>.py                : RL environments (physics simulations, interfaces to experiments, etc.)
| ├── workflows      	                : folder containing ExaRL workflows and registration scripts
|     └── __init__.py                   : workflow registry
|     └── registration.py               : script to handle registration
|     ├── workflow_vault                : folder containing workflows
|         └── __init__.py               : script to make workflows visible
|         ├── workflow_cfg              : folder containing default workflow configurations
|         └── <RLworkflow>.py           : RL workflows (such as SEED, IMPALA, etc.)
| ├── utils                             : folder containing utilities
|     └── __init__.py                   : make classes and functions visible
|     └── candleDriver.py               : Supporting CANDLE script
|     └── analyze_reward.py             : script for plotting results
|     └── log.py                        : central place to set logging levels
|     └── profile.py                    : provides function decorators for profiling, timing, and debugging