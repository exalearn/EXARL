import driver.candleDriver as cd

run_params = cd.initialize_parameters()
env = run_params['env']

if env == 'ExaLearnBlockCoPolymerTDLG-v0':
    from envs.env_vault.exalearn_bcp_tdlg import BlockCoPolymerTDLG
elif env == 'ExaLearnCartpole-v0':
    from envs.env_vault.ExaCartpoleDynamic import ExaCartpoleDynamic
elif env == 'ExaLearnCartpole-v1':
    from envs.env_vault.ExaCartpoleStatic import ExaCartpoleStatic
elif env == 'ch-v0':
    from envs.env_vault.env_ch import CahnHilliardEnv
elif env == 'ExaCovid-v0':
    from envs.env_vault.ExaCOVID import ExaCOVID

