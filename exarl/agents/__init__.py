from exarl.agents.registration import register, make
import exarl.utils.candleDriver as cd
agent = cd.lookup_params('agent')

if agent == 'DQN-v0':
    register(
        id=agent,
        entry_point='exarl.agents.agent_vault:DQN'
    )

if agent == 'DQN-v1':
    register(
        id=agent,
        entry_point='exarl.agents.agent_vault:DQNparallel'
    )

elif agent == 'DDPG-v0':
    register(
        id=agent,
        entry_point='exarl.agents.agent_vault:DDPG'
    )

elif agent == 'DDPG-v4':
    register(
        id=agent,
        entry_point='exarl.agents.agent_vault:DDPG_V4'
    )
elif agent == 'DDDQN-v0':
    register(
        id=agent,
        entry_point='exarl.agents.agent_vault:DDPGvtrace'
    )

elif agent == 'DDPG-v2':
    register(
        id=agent,
        entry_point='exarl.agents.agent_vault:DDPG_LSTM'
    )

elif agent == 'DDPG-v3':
    register(
        id=agent,
        entry_point='exarl.agents.agent_vault:DDPG_PER'
    )

elif agent == 'TD3-v0':
    register(
        id=agent,
        entry_point='exarl.agents.agent_vault:TD3'
    )
elif agent == 'SAC-v0':
    register(
        id=agent,
        entry_point='exarl.agents.agent_vault:SAC'
    )

elif agent == 'DDDQN-v0':
    register(
        id=agent,
        entry_point='exarl.agents.agent_vault:DDPG_Vtrace'
    )

elif agent == 'A2C-v0':
    register(
        id=agent,
        entry_point='exarl.agents.agent_vault:A2C'
    )
elif agent == 'A2C-v1':
    register(
        id=agent,
        entry_point='exarl.agents.agent_vault:A2Cvtrace'
    )
elif agent == 'A2C-v2':
    register(
        id=agent,
        entry_point='exarl.agents.agent_vault:A2Ccontinuous'
    )
else:
    print("No agent selected!")
