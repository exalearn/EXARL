from gym.envs import registration
from gym.envs.registration import register
import utils.candleDriver as cd


env = cd.run_params['env']

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
        id=env.lower(),
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

elif env == 'ExaBoosterDiscrete-v0':
    register(
        id=env,
        entry_point='envs.env_vault:ExaBooster'
    )
    
elif env == 'ExaWaterClusterDiscrete-v0':
    register(
        id=env,
        entry_point='envs.env_vault:ExaWaterClusterDiscrete'
    )
