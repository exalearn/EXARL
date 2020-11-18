from gym.envs import registration
from gym.envs.registration import register
from utils.candleDriver import initialize_parameters


run_params = initialize_parameters()
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
        kwargs={"app_dir":'./envs/env_vault/LibTDLG'},
    )
    
elif env == 'ExaLearnCartPole-v0':
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

elif env == 'ExaBooster-v1':
    register(
        id=env,
        entry_point='envs.env_vault:ExaBooster'
    )

elif env == 'ExaLAMMPS-v0':
    register(
        id=env,
        entry_point='envs.env_vault:ExaLAMMPS'
    )

elif env == 'ExaLearnWaterCluster-v0':
    register(
        id=env,
        entry_point='envs.env_vault:WaterCluster'
    )
