from exa_rl.exa_learner import ExaLearner

## Define agent and env
agent_id = 'exa_agents:DQN-v0'
env_id   = 'CartPole-v0'

## Create ExaLearner
exa_learner = ExaLearner(agent_id,env_id)
exa_learner.set_results_dir('./exa_learner_example1_results/')
exa_learner.set_training(100,100)
exa_learner.run()
