import gym
import exa_envs
import exa_agents

def make(agent_id, env_id):
	env = gym.make(env_id)
	agent = exa_agents.make(agent_id, env=env)

	return agent, env	
