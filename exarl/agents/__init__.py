from exarl.agents.registration import register, make
import exarl.utils.candleDriver as cd
agent = cd.lookup_params('agent')

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
elif agent == 'DDPG-VTRACE-v0':
    register(
        id=agent,
        entry_point='exarl.agents.agent_vault:DDPG_Vtrace'
    )
elif agent == 'A2C-v0':
    register(
        id=agent,
        entry_point='exarl.agents.agent_vault:A2C'
    )
else:
    print("No agent selected!")
