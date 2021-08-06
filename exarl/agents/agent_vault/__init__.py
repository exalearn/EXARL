import exarl.utils.candleDriver as cd

agent = cd.run_params['agent']

if agent == 'DQN-v0':
    from exarl.agents.agent_vault.dqn import DQN
elif agent == 'DQN-v1':
    from exarl.agents.agent_vault.dqn_parallel import DQNparallel
elif agent == 'DDPG-v0':
    from exarl.agents.agent_vault.ddpg import DDPG
elif agent == 'DDPG-v1':
    from exarl.agents.agent_vault.ddpg_vtrace import DDPGvtrace
elif agent == 'DDPG-v2':
    from exarl.agents.agent_vault.ddpg_lstm import DDPG_LSTM
elif agent == 'DDPG-v3':
    from exarl.agents.agent_vault.ddpg_per import DDPG_PER
elif agent == 'TD3-v0':
    from exarl.agents.agent_vault.td3 import TD3
elif agent == 'DDPG-v4':
    from exarl.agents.agent_vault.ddpg_v4 import DDPG_V4
elif agent == 'DDDQN-v0':
    from exarl.agents.agent_vault.dddqn import DDDQN
elif agent == 'MLDQN-v0':
    from exarl.agents.agent_vault.mldqn import MLDQN
elif agent == 'A2C-v0':
    from exarl.agents.agent_vault.a2c import A2C
elif agent == 'A2C-v1':
    from exarl.agents.agent_vault.a2c_vtrace import A2Cvtrace
elif agent == 'A2C-v2':
    from exarl.agents.agent_vault.a2c_continuous import A2Ccontinuous
elif agent == 'SAC-v0':
    from exarl.agents.agent_vault.sac import SAC
elif agent == 'DQN-v1':
    from exarl.agents.agent_vault.dqnher import DQNHER
