import exarl as erl
import time

## Define agent and env
agent_id = 'agents:DQN-v0'
env_id   = 'envs:ExaLearnCartpole-v0'

## Create ExaDQN
exa_dqn = erl.ExaLearner(agent_id,env_id)
exa_dqn.set_results_dir('./exa_dqn_results/')
exa_dqn.set_training(10,10)
start = time.time()
exa_dqn.run()
stop = time.time()
print("Elapsed time = ", stop - start)
