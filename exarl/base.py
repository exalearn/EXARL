import json

class base():
	def __init__(self, **kwargs):
                for key, value in kwargs.items():
                        if key == 'env_cfg':
                                self.env_cfg = value
                        else:
                                self.env_cfg = 'envs/env_vault/env_cfg/env_setup.json'
                                
                        if key == 'agent_cfg':
                                self.agent_cfg = value
                        else:
                                self.agent_cfg = 'agents/agent_vault/agent_cfg/dqn_setup.json'

                with open(self.env_cfg) as json_file:
                        env_data = json.load(json_file)

                self.num_child_per_parent = int(env_data['child_spawn_per_parent']) if 'child_spawn_per_parent' in env_data.keys() else 0
                if(self.num_child_per_parent > 0):
                        # defaults to running toy example of computing PI
                        self.worker = (env_data['worker_app']).lower() if 'worker_app' in env_data.keys() else "envs/env_vault/cpi.py"
                else:
                        self.worker = None

                with open(self.agent_cfg) as json_file:
                        agent_data = json.load(json_file)

                self.search_method =  (agent_data['search_method']).lower() if 'search_method' in agent_data.keys() else "epsilon"  # discount rate
                self.gamma =  float(agent_data['gamma']) if 'gamma' in agent_data.keys() else 0.95  # discount rate
                self.epsilon = float(agent_data['epsilon']) if 'epsilon' in agent_data.keys() else 1.0  # exploration rate
                self.epsilon_min = float(agent_data['epsilon_min']) if 'epsilon_min' in agent_data.keys() else 0.05
                self.epsilon_decay = float(agent_data['epsilon_decay']) if 'epsilon_decay' in agent_data.keys() else 0.995
                self.learning_rate =  float(agent_data['learning_rate']) if 'learning_rate' in agent_data.keys() else  0.001
                self.batch_size = int(agent_data['batch_size']) if 'batch_size' in agent_data.keys() else 32
                self.tau = float(agent_data['tau']) if 'tau' in agent_data.keys() else 0.5
