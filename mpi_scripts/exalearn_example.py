import exarl as erl
import time

## Define agent and env
agent_id = 'agents:DQN-v0'
env_id   = 'envs:ExaLearnCartpole-v1' # Use ExaLearnCartpole-v0 for dynamic version

## Create ExaDQN
exa_learner = erl.ExaLearner(agent_id,env_id)
exa_learner.set_results_dir('./exa_dqn_results/')
exa_learner.set_training(100,100)
start = time.time()
exa_learner.run('static-multi-groups') # can be either 'static' or 'dynamic'
stop = time.time()
print("Elapsed time = ", stop - start)
