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