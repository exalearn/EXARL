from gym.envs.registration import register

#register(
#    id='ch-v0',
#    entry_point='envs:CahnHilliardEnv',
#)    

register(
    id='ExaLearnBlockCoPolymerTDLG-v0',
    entry_point='exa_envs.envs:BlockCoPolymerTDLG',
    kwargs={'cfg_file': 'exa_envs/envs/env_cfg/tdlg_setup.json'}
)

register(
    id='ExaLearnCartpole-v0',
    entry_point='exa_envs.envs:ExaCartpole'
)
