from exarl.utils.globals import ExaGlobals
try:
    env = ExaGlobals.lookup_params('env')
except:
    env = None

if env == 'ExaCartPoleStatic-v0':
    from exarl.envs.env_vault.ExaCartpoleStatic import ExaCartpoleStatic
elif env == 'ExaCH-v0':
    from exarl.envs.env_vault.ExaCH import CahnHilliardEnv
elif env == 'ExaCovid-v0':
    from exarl.envs.env_vault.ExaCOVID import ExaCOVID
elif env == 'ExaBoosterDiscrete-v0':
    from exarl.envs.env_vault.ExaBoosterDiscrete import ExaBooster_v1 as ExaBooster
elif env == 'ExaBoosterNew-v0':
    from exarl.envs.env_vault.ExaBoosterNew import ExaBooster_v2 as ExaBooster
elif env == 'ExaWaterClusterDiscrete-v0':
    from exarl.envs.env_vault.ExaWaterClusterDiscrete import ExaWaterClusterDiscrete
elif env == 'Hadrec-v0':
    from exarl.envs.env_vault.HadrecWrapper import HadrecWrapper
elif env == 'Hadrec-v1':
    from exarl.envs.env_vault.HadrecWrapper_V1 import HadrecWrapper_V1
elif env == 'Bsuite-v0':
    from exarl.envs.env_vault.BsuiteWrapper import BsuiteWrapper
elif env == 'MyPendulum-v0':
    from exarl.envs.env_vault.MyPendulum import MyPendulum
