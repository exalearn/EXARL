from gym.envs import registration
from gym.envs.registration import register
import exarl.utils.candleDriver as cd

env = cd.run_params['env']

if env == 'ExaCH-v0':
    register(
        id=env,
        entry_point='exarl.envs.env_vault:CahnHilliardEnv',
    )

elif env == 'ExaCartPoleStatic-v0':
    register(
        id=env,
        entry_point='exarl.envs.env_vault:ExaCartpoleStatic'
    )

elif env == 'ExaCovid-v0':
    register(
        id=env,
        entry_point='exarl.envs.env_vault:ExaCOVID'
    )

elif env == 'ExaBoosterDiscrete-v0':
    register(
        id=env,
        entry_point='exarl.envs.env_vault:ExaBooster'
    )

elif env == 'ExaWaterClusterDiscrete-v0':
    register(
        id=env,
        entry_point='exarl.envs.env_vault:ExaWaterClusterDiscrete'
    )

elif env == 'GymSpaceTest-v0':
    register(
        id=env,
        entry_point='exarl.envs.env_vault:GymSpaceTest'
    )

elif env == 'Bsuite-v0':
    register(
        id=env,
        entry_point='exarl.envs.env_vault:BsuiteWrapper'
    )

elif env == 'Marshmallow-v0':
    register(
        id=env,
        entry_point='exarl.envs.env_vault:Marshmallow'
    )