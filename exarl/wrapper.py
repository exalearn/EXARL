import gym
import envs
import agents

def make(agent_id, env_id):
	env = gym.make(env_id)
	agent = agents.make(agent_id, env=env)

	return agent, env	
