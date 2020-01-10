rom exa_base.exa_dqn import ExaDQN

## Define agent and env
agent_id = 'exa_agents:DQN-v0'
env_id   = 'exa_envs:ExaLearnCartpole-v0'

## Create ExaDQN
exa_dqn = ExaDQN(agent_id,env_id)
exa_dqn.set_results_dir('./exa_dqn_results/')
exa_dqn.set_training(10,10)
exa_dqn.run()