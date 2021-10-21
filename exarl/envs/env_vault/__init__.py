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
elif env == 'ExaWaterClusterDiscrete-v0':
    from exarl.envs.env_vault.ExaWaterClusterDiscrete import ExaWaterClusterDiscrete
elif env == 'GymSpaceTest-v0':
    from exarl.envs.env_vault.GymSpaceTest import GymSpaceTest
# TODO: Make more general for any bsuite env. Currently using one bandit problem.
elif env == 'ExaBsuite-v0':
    from exarl.envs.env_vault.ExaBsuite import ExaBsuite
