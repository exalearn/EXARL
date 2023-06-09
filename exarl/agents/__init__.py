from exarl.agents.registration import register, make
from exarl.utils.globals import ExaGlobals
try:
    agent = ExaGlobals.lookup_params('agent')
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
elif agent == 'GraphTD3-v0':
    register(
        id=agent,
        entry_point='exarl.agents.agent_vault:KerasGraphTD3'
    )
elif agent == 'PARS-v0':
    register(
        id=agent,
        entry_point='exarl.agents.agent_vault:PARS'
    )
