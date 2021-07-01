import exarl.utils.candleDriver as cd

agent = cd.run_params['agent']

if agent == 'DQN-v0':
    from exarl.agents.agent_vault.dqn import DQN
elif agent == 'DDPG-v0':
    from exarl.agents.agent_vault.ddpg import DDPG
elif agent == 'DDDQN-v0':
    from exarl.agents.agent_vault.dddqn import DDDQN
elif agent == 'MLDQN-v0':
    from exarl.agents.agent_vault.mldqn import MLDQN
elif agent == 'A2C-v0':
    from exarl.agents.agent_vault.a2c import A2C
elif agent == 'A2C-v1':
    from exarl.agents.agent_vault.a2c_vtrace import A2Cvtrace
