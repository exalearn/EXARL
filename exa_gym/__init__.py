from gym.envs.registration import register

register(
    id='ch-v0',
    entry_point='envs:CahnHilliardEnv',
)    
