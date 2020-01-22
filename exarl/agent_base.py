import json
from abc import ABC, abstractmethod

class ExaAgent(ABC):

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if key == 'agent_cfg':
                self.agent_cfg = value
            else:
                self.agent_cfg = 'agents/agent_vault/agent_cfg/dqn_setup.json'

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

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def action(self):
        pass

    @abstractmethod
    def remember(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def monitor(self):
        pass
