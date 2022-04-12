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
