import utils.candleDriver as cd

env = cd.run_params['env']

if env == 'ExaCartPole-v1':
    from envs.env_vault.ExaCartpoleStatic import ExaCartpoleStatic
elif env == 'ExaCH-v0':
    from envs.env_vault.ExaCH import CahnHilliardEnv
elif env == 'ExaCovid-v0':
    from envs.env_vault.ExaCOVID import ExaCOVID
<<<<<<< HEAD
elif env == 'ExaBooster-v1':
    from envs.env_vault.surrogate_accelerator_v1 import Surrogate_Accelerator_v1 as ExaBooster
elif env == 'ExaBoosterContinuous-v1':
    from envs.env_vault.surrogate_accelerator_v2 import Surrogate_Accelerator_v2 as ExaBoosterContinuous
elif env == 'ExaLAMMPS-v0':
    from envs.env_vault.exalearn_lammps_ex1 import ExaLammpsEx1 as ExaLAMMPS
elif env == 'ExaWaterCluster-v0':
    from envs.env_vault.exalearn_water_cluster import WaterCluster
elif env == 'ExaWaterClusterContinuous-v0':
    from envs.env_vault.exalearn_water_cluster_cont import WaterCluster
=======
elif env == 'ExaBoosterDiscrete-v0':
    from envs.env_vault.ExaBoosterDiscrete import ExaBooster_v1 as ExaBooster
elif env == 'ExaWaterClusterDiscrete-v0':
    from envs.env_vault.ExaWaterClusterDiscrete import ExaWaterClusterDiscrete
>>>>>>> 201e3962c07b29dd132565f115b53932bdabdce1
