from gym.envs.registration import register
import driver.candleDriver as cd

run_params = cd.initialize_parameters()
env = run_params['env']

if env == 'ch-v0':
    register(
        id=env,
        entry_point='envs.env_vault:CahnHilliardEnv',
    )

elif env == 'ExaLearnBlockCoPolymerTDLG-v0':
    register(
        id=env,
        entry_point='envs.env_vault:BlockCoPolymerTDLG',
    )
    
elif env == 'ExaLearnBlockCoPolymerTDLG-v3':
    register(
        id=env,
        entry_point='envs.env_vault:BlockCoPolymerTDLGv3',
    )
    
elif env == 'ExaLearnCartpole-v0':
    register(
        id=env,
        entry_point='envs.env_vault:ExaCartpoleDynamic'
    )

elif env == 'ExaLearnCartpole-v1':
    register(
        id=env,
        entry_point='envs.env_vault:ExaCartpoleStatic'
    )

elif env == 'ExaCovid-v0':
    register(
        id=env,
        entry_point='envs.env_vault:ExaCOVID'
    )
