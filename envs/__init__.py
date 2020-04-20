from gym.envs.registration import register

#register(
#    id='ch-v0',
#    entry_point='envs:CahnHilliardEnv',
#)    

register(
    id='ExaLearnBlockCoPolymerTDLG-v0',
    entry_point='envs.env_vault:BlockCoPolymerTDLG',
    kwargs={'cfg_file': 'envs/env_vault/env_cfg/tdlg_setup.json'}
)

register(
    id='ExaLearnCartpole-v0',
    entry_point='envs.env_vault:ExaCartpoleDynamic'
)

register(
    id='ExaLearnCartpole-v1',
    entry_point='envs.env_vault:ExaCartpoleStatic'
)
