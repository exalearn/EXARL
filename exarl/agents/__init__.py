from exarl.agents.registration import register, make
import exarl.utils.candleDriver as cd
try:
    agent = cd.run_params['agent']
except:
    agent = None

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
elif agent == 'TD3-v0':
    register(
        id=agent,
        entry_point='exarl.agents.agent_vault:TD3'
    )
elif agent == 'TD3-v1':
    register(
        id=agent,
        entry_point='exarl.agents.agent_vault:KerasTD3'
    )
elif agent == 'PARS-v0':
    register(
        id=agent,
        entry_point='exarl.agents.agent_vault:PARS'
    )
