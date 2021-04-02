import utils.candleDriver as cd

env = cd.run_params['env']

if env == 'ExaCartPole-v0':
    from envs.env_vault.ExaCartpoleDynamic import ExaCartpoleDynamic
elif env == 'ExaCartPole-v1':
    from envs.env_vault.ExaCartpoleStatic import ExaCartpoleStatic
elif env == 'ExaCH-v0':
    from envs.env_vault.ExaCH import CahnHilliardEnv
elif env == 'ExaCovid-v0':
    from envs.env_vault.ExaCOVID import ExaCOVID
elif env == 'ExaBoosterDiscrete-v0':
    from envs.env_vault.ExaBoosterDiscrete import ExaBooster_v1 as ExaBooster
elif env == 'ExaWaterClusterDiscrete-v0':
    from envs.env_vault.ExaWaterClusterDiscrete import ExaWaterClusterDiscrete
