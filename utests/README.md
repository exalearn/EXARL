# Unit Testing for ExaRL
A pytest based unit testing is implemented to evaluate ExaRL.  Our current testing strategy is focused on the base classes of ExaRL 
including environments, agents, workflows, communicators, and data structures.

## Unit Test Cases for DQN agent
The pytest framework allows for tests to be divided by file, class, and individual test.
We spread our unit tests across the following files:
* utest_comm.py : This module tests the basic communicator wrappers.  It focuses on class members and comm splitting.
* utest_data_structure.py : This module test RMA data structures for correctness.  There are some performance tests which can be manually tuned (they are turned off by default).
* utest_env.py : This module can tests the functionality set by openAI gym that is required by ExaRL.
* utest_agent.py : This module tests various members of the ExaRL agent.  It checks for required methods, it does not check for correct learning.
* utest_workflow.py : This module tests for correct functionality of a workflow.

utest_env.py and utest_agent.py designed to help users of ExaRL create correct environment and agents.  Each test has configurable arguments which can be set. 

For utest_env.py: 
* test_env_name - gym name for test (e.g. ExaCartPoleStatic-v0)
* test_env_class - name of the class for test module (e.g. ExaCartpoleStatic)
* test_env_file - name of the file containing test_env_class omitting the ".py" (e.g. ExaCartpoleStatic)
To use call:
```
./utest_env.py --test_env_name ExaCartPoleStatic-v0 --test_env_class ExaCartpoleStatic --test_env_file ExaCartpoleStatic
```
If only test_env_name is given, we assume the environment is already in the gym registry. If no arguments are given a synthetic environment is generated.

For utest_agent.py:
This is a pytest fixture to add an agent to the agent registry based on command line arguments.
* test_agent_name - gym name for test (e.g. DQN-v0)
* test_agent_class - name of the class for test module (e.g. DQN)
* test_agent_file - name of the file containing test_env_class omitting the ".py" (e.g. dqn)
* test_env_name - gym name for test (e.g. ExaCartPoleStatic-v0)
* test_env_class - name of the class for test module (e.g. ExaCartpoleStatic)
* test_env_file - name of the file containing test_env_class omitting the ".py" (e.g. ExaCartpoleStatic)
* test_save_load_dir - this is the path to a directory to use for testing saving and loading weights
To use call: 
```
pytest ./utest_agent.py --test_agent_name DQN-v0 --test_agent_class DQN --test_agent_file dqn
```
If the environment parameters are omitted a synthetic environment is generated.

There are additional flags which can be used for utest_workflow.py
* on-policy - configures how off policy an actor can be before asserting.  Set to -1 to just record (default).
* behind - configures how old of data a learner can accept from an actor before asserting.  Set to -1 to just record (default).
* rank_sleep - Toggles on/off rank based sleeping scheme for training and stepping.  Default is off.
* random_sleep - Toggles on/off random sleeping for training and stepping.  Default is off.
To use call:
```
pytest ./utest_workflow.py --on-policy 1 --behind 1 --random_sleep 
```

## Run
There are many ways to run pytest.  The following is a helpful guide:
https://docs.pytest.org/en/7.1.x/how-to/usage.html

Batch schedulers wrap nicely around pytest:
```
srun -N 1 -n 2  pytest utests/utest_workflow.py
```
The current test will try to create various configurations of leaners and actors based on the number of ranks provided.  If a current node count is not support within a test, it will skip all the tests.  This will happen for the utest_env.py test if invoked with only a single rank.  All other tests can be run with a single rank, but care should be taken to make sure it is meaningful.  For example testing an async learner with only one rank would be problematic.

## pytest.ini
The test methods (test_*) can be executed by the pytest framework by running 'pytest' command from the ExaRL parent directory (exarl/).
```
ExaRL/utests % cd ..
ExaRL % pytest
```
The pytest command looks for the pytest.ini file in ExaRL/
The pytest.ini file is a configuration file used by the pytest framework. It includes command-line parameters and flags, which is specified by 'addopts'
```
pytest.ini file

addopts = --ignore=./envs --showlocals --color=yes
```
Other configuration parameters in pytest.ini are:
* python_files: It identifies *.py files which are only run by pytest command.
* python_functions: It identifies the specific test functions only to be run inside a 'python_files' file.
* testpaths: It specifies folder names in ExaRL/ which are the only folders run by the pytest command.
* Other configurations are dedicated for logging.

## Integration with Travis CI
The pytest framework for unit testing has been integrated with build test framework provided by Travis CI. Consequently, ExaRL/.travis.yml and ExaRL/setup.py files have been updated to take effect.
