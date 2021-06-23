import exarl.utils.candleDriver as cd
agent = cd.lookup_params('agent')

if agent == 'DQN-v0':
    from exarl.agents.agent_vault.dqn import DQN
elif agent == 'DDPG-v0':
    from exarl.agents.agent_vault.ddpg import DDPG
elif agent == 'DDPG-VTRACE-v0':
    from exarl.agents.agent_vault.ddpg_vtrace import DDPG_Vtrace
elif agent == 'DDDQN-v0':
    from exarl.agents.agent_vault.dddqn import DDDQN
elif agent == 'MLDQN-v0':
    from exarl.agents.agent_vault.mldqn import MLDQN
elif agent == 'A2C-v0':
    from exarl.agents.agent_vault.a2c import A2C
else:
    print("No agent selected!")
