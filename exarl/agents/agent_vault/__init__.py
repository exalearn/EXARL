import exarl.utils.candleDriver as cd
agent = cd.lookup_params('agent')

if agent == 'DQN-v0':
    from exarl.agents.agent_vault.dqn import DQN
elif agent == 'DDPG-v0':
    from exarl.agents.agent_vault.ddpg import DDPG
elif agent == 'DDPG-VTRACE-v0':
    from exarl.agents.agent_vault.ddpg_vtrace import DDPG_Vtrace
elif agent == 'TD3-v0':
    from exarl.agents.agent_vault.td3 import TD3
else:
    print("No agent selected!")
