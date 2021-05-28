from exarl.agents.registration import register, make
import exarl.utils.candleDriver as cd

agent = cd.run_params['agent']

if agent == 'DQN-v0':
    register(
        id=agent,
        entry_point='exarl.agents.agent_vault:DQN'
    )
elif agent == 'DDPG-v0':
    register(
        id=agent,
        entry_point='exarl.agents.agent_vault:DDPG'
    )
elif agent == 'DDDQN-v0':
    register(
        id=agent,
        entry_point='exarl.agents.agent_vault:DDDQN'
    )
elif agent == 'MLDQN-v0':
    register(
        id=agent,
        entry_point='exarl.agents.agent_vault:MLDQN'
    )
