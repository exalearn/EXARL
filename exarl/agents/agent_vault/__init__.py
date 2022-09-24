from exarl.utils.globals import ExaGlobals
try:
    agent = ExaGlobals.lookup_params('agent')
except:
    agent = None

if agent == 'DQN-v0':
    from exarl.agents.agent_vault.dqn import DQN
elif agent == 'DDPG-v0':
    from exarl.agents.agent_vault.ddpg import DDPG
elif agent == 'DDPG-VTRACE-v0':
    from exarl.agents.agent_vault.ddpg_vtrace import DDPG_Vtrace
elif agent == 'TD3-v0':
    from exarl.agents.agent_vault.td3 import TD3
elif agent == 'TD3-v1':
    from exarl.agents.agent_vault.keras_td3 import KerasTD3
elif agent == 'PARS-v0':
    from exarl.agents.agent_vault.PARS import PARS
elif agent == 'BSUITE-BASE-v0':
    from exarl.agents.agent_vault.bsuite_base import serial_bsuite_agent
elif agent == 'BSUITE-BASE-v1':
    from exarl.agents.agent_vault.bsuite_base import parallel_bsuite_agent
elif agent == 'TORCH-AGENT-DQN-v0':
    from exarl.agents.agent_vault.torch_agents.dqn_wrapper import torch_dqn
elif agent == 'TORCH-AGENT-PPO-v0':
    from exarl.agents.agent_vault.torch_agents.ppo_wrapper import torch_ppo
elif agent == 'TORCH-AGENT-A3C-v0':
    from exarl.agents.agent_vault.torch_agents.a3c_wrapper import torch_a3c
elif agent == 'EXA-A2C-v0':
    from exarl.agents.agent_vault.exa_a2c import A2C