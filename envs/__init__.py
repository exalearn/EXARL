from gym.envs.registration import register

register(
    id='ExaLearnCartpole-v0',
    entry_point='envs.env_vault:ExaCartpoleDynamic'
)

register(
    id='ExaLearnCartpole-v1',
    entry_point='envs.env_vault:ExaCartpoleStatic'
)

register(
    id='ExaCovid-v0',
    entry_point='envs.env_vault:ExaCOVID'
)
