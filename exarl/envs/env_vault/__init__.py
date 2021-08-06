import exarl.utils.candleDriver as cd

env = cd.run_params['env']

if env == 'ExaCartPoleStatic-v0':
    from exarl.envs.env_vault.ExaCartpoleStatic import ExaCartpoleStatic
elif env == 'ExaCH-v0':
    from exarl.envs.env_vault.ExaCH import CahnHilliardEnv
elif env == 'ExaCovid-v0':
    from exarl.envs.env_vault.ExaCOVID import ExaCOVID
elif env == 'ExaBoosterDiscrete-v0':
    from exarl.envs.env_vault.ExaBoosterDiscrete import ExaBooster_v1 as ExaBooster
<<<<<<< HEAD
elif env == 'ExaWaterClusterDiscrete-v0':
    from exarl.envs.env_vault.ExaWaterClusterDiscrete import ExaWaterClusterDiscrete
elif env == 'ExaParabola-v0':
    from exarl.envs.env_vault.ExaParabola import ExaParabola
elif env == 'ExaParabolaContinuous-v0':
    from exarl.envs.env_vault.ExaParabolaContinuous import ExaParabolaContinuous
