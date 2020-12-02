from gym.envs import registration
from gym.envs.registration import register
from utils.candleDriver import initialize_parameters


run_params = initialize_parameters()
env = run_params['env']


if env == 'ExaCH-v0':
    register(
        id=env,
        entry_point='envs.env_vault:CahnHilliardEnv',
    )

elif env == 'ExaTDLG-v0':
    register(
        id=env,
        entry_point='envs.env_vault:BlockCoPolymerTDLG',
    )

elif env == 'ExaTDLG-v3':
    register(
        id=env,
        entry_point='envs.env_vault:BlockCoPolymerTDLGv3',
        kwargs={"app_dir": './envs/env_vault/LibTDLG'},
    )

elif env == 'ExaCartPole-v0':
    register(
        id=env,
        entry_point='envs.env_vault:ExaCartpoleDynamic'
    )

elif env == 'ExaCartPole-v1':
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

elif env == 'ExaBoosterContinuous-v1':
    register(
        id=env,
        entry_point='envs.env_vault:ExaBoosterContinuous'
    )

elif env == 'ExaLAMMPS-v0':
    register(
        id=env,
        entry_point='envs.env_vault:ExaLAMMPS'
    )
elif env == 'ExaWaterCluster-v0':
    register(
        id=env,
        entry_point='envs.env_vault:WaterCluster'
    )
