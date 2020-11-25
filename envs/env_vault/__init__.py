from utils.candleDriver import initialize_parameters

run_params = initialize_parameters()
env = run_params['env']

if env == 'ExaTDLG-v0':
    from envs.env_vault.exalearn_bcp_tdlg import BlockCoPolymerTDLG
elif env == 'ExaTDLG-v3':
    from envs.env_vault.exalearn_bcp_tdlg_v3 import BlockCoPolymerTDLGv3
elif env == 'ExaCartPole-v0':
    from envs.env_vault.ExaCartpoleDynamic import ExaCartpoleDynamic
elif env == 'ExaCartPole-v1':
    from envs.env_vault.ExaCartpoleStatic import ExaCartpoleStatic
elif env == 'ExaCH-v0':
    from envs.env_vault.env_ch import CahnHilliardEnv
elif env == 'ExaCovid-v0':
    from envs.env_vault.ExaCOVID import ExaCOVID
elif env == 'ExaBooster-v1':
    from envs.env_vault.surrogate_accelerator_v1 import Surrogate_Accelerator_v1 as ExaBooster
elif env == 'ExaBoosterContinuous-v1':
    from envs.env_vault.surrogate_accelerator_v2 import Surrogate_Accelerator_v2 as ExaBoosterContinuous
elif env == 'ExaLAMMPS-v0':
    from envs.env_vault.exalearn_lammps_ex1 import ExaLammpsEx1 as ExaLAMMPS
if env == 'ExaDotsAndBoxes-v0':
    from envs.env_vault.ExaDotsAndBoxes import ExaDotsAndBoxes
if env == 'GymSpaceTest-v0':
    from envs.env_vault.GymSpaceTest import GymSpaceTest
