import exarl.utils.candleDriver as cd
env = cd.lookup_params('env')

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
