from gym.envs import registration
from gym.envs.registration import register
from exarl.utils.globals import ExaGlobals
try:
    env = ExaGlobals.lookup_params('env')
except:
    env = None

if env == 'ExaCH-v0':
    register(
        id=env,
        entry_point='exarl.envs.env_vault:CahnHilliardEnv',
    )

elif env == 'ExaExaaltGraphConstrained-v0':
    register(
        id=env,
        entry_point='exarl.envs.env_vault:ExaExaaltGraphConstrained'
    )

elif env == 'ExaExaaltSimple-v0':
    register(
        id=env,
        entry_point='exarl.envs.env_vault:ExaExaaltSimple'
    )

elif env == 'ExaExaaltBayesRL-v0':
    register(
        id=env,
        entry_point='exarl.envs.env_vault:ExaExaaltBayesRL'
    )

elif env == 'ExaExaaltBayesRLSparse-v0':
    register(
        id=env,
        entry_point='exarl.envs.env_vault:ExaExaaltBayesRLSparse'
    )

elif env == 'ExaExaaltVE-v0':
    register(
        id=env,
        entry_point='exarl.envs.env_vault:ExaExaaltVE'
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
