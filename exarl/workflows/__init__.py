from exarl.workflows.registration import register, make

register(
    id='sync',
    entry_point='exarl.workflows.workflow_vault:SYNC'
)

register(
    id='sync2',
    entry_point='exarl.workflows.workflow_vault:SYNC2'
)

register(
    id='async',
    entry_point='exarl.workflows.workflow_vault:ASYNC'
)

register(
    id='async2',
    entry_point='exarl.workflows.workflow_vault:ASYNC2'
)

register(
    id='async_parallel',
    entry_point='exarl.workflows.workflow_vault:ASYNCparallel'
)

register(
    id='rma',
    entry_point='exarl.workflows.workflow_vault:RMA_ASYNC'
)

register(
    id='rma_v2',
    entry_point='exarl.workflows.workflow_vault:RMA_ASYNC_v2'
)

register(
    id='mlrma',
    entry_point='exarl.workflows.workflow_vault:ML_RMA'
)

register(
    id='seed',
    entry_point='exarl.workflows.workflow_vault:SEED'
)
