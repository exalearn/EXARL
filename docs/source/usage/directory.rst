Directory Organization
**********************

| ├── setup.py                          : Python setup file with requirements files
| ├── scripts                           : folder containing RL steering scripts
| ├── exarl                	            : folder with EXARL code
|     └── __init__.py                   : make base classes visible
|     └── mpi_settings.py               : MPI settings
|     ├── base         	                : folder containing EXARL base classes
|         └── __init__.py               : make base classes visible
|         └── agent_base.py             : agent base class
|         └── env_base.py               : environment base class
|         └── workflow_base.py          : workflow base class
|         └── learner_base.py           : learner base class
|     ├── config                	      : folder containing configurations
|         └── agent_cfg                 : agent configuration folder
|         └── model_cfg                 : model configuration folder
|         └── env_cfg                   : env configuration folder
|         └── workflow_cfg              : workflow configuration folder
|         └── learner_cfg.json          : learner configuration
|     ├── driver                        : folder containing RL MPI steering scripts
|         └── driver.py                 : Run scipt
|     ├── candlelib                     : folder containing library for CANDLE functionality
|     ├── agents         	              : folder containing EXARL agents and registration scripts
|         └── __init__.py               : agent registry
|         └── registration.py           : script to handle registration
|         ├── agent_vault               : folder containing agents
|             └── __init__.py           : script to make agents visible
|             └── <RLagent>.py          : RL agents (such as DQN, DDPG, etc.)
|     ├── envs         	                : folder containing EXARL environments
|         └── __init__.py               : environment registry
|         ├── env_vault                 : folder containing environments
|         └── __init__.py               : script to make environments visible
|             └── <RLenv>.py            : RL environments (physics simulations, interfaces to experiments, etc.)
|     ├── workflows      	              : folder containing EXARL workflows and registration scripts
|         └── __init__.py               : workflow registry
|         └── registration.py           : script to handle registration
|         ├── workflow_vault            : folder containing workflows
|             └── __init__.py           : script to make workflows visible
|             └── <RLworkflow>.py       : RL workflows (such as SEED, IMPALA, etc.)
|     ├── utils                         : folder containing utilities
|         └── __init__.py               : make classes and functions visible
|         └── candleDriver.py           : Supporting CANDLE script
|         └── analyze_reward.py         : script for plotting results
|         └── log.py                    : central place to set logging levels
|         └── profile.py                : provides function decorators for profiling, timing, and debugging