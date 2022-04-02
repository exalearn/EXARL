from gym.envs import registration
from gym.envs.registration import register
import exarl.utils.candleDriver as cd
env = cd.lookup_params('env')

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

elif env == 'ExaBoosterNew-v0':
    register(
        id=env,
        entry_point='exarl.envs.env_vault:ExaBooster'
    )

elif env == 'ExaWaterClusterDiscrete-v0':
    register(
        id=env,
        entry_point='exarl.envs.env_vault:ExaWaterClusterDiscrete'
    )

elif env == 'Hadrec-v0':
    register(
        id=env,
        entry_point='exarl.envs.env_vault:HadrecWrapper'
    )

elif env == 'Bsuite-v0':
    register(
        id=env,
        entry_point='exarl.envs.env_vault:BsuiteWrapper'
    )
