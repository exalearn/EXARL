import exarl as erl
import time

## Define agent and env
agent_id = 'agents:DQN-v0'
env_id   = 'envs:ExaLearnCartpole-v1' # Use ExaLearnCartpole-v0 for dynamic version

## Create ExaDQN
exa_learner = erl.ExaLearner(agent_id,env_id)
exa_learner.set_results_dir('./exarl_results_dir/')
exa_learner.set_training(10,10)
start = time.time()
run_type = exa_learner.env.run_type
exa_learner.run(run_type) # can be either 'static' or 'dynamic'
stop = time.time()
print("Elapsed time = ", stop - start)
