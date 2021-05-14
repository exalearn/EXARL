from exarl.workflows.registration import register, make

register(
    id='sync',
    entry_point='exarl.workflows.workflow_vault:SYNC'
)

register(
    id='async',
    entry_point='exarl.workflows.workflow_vault:ASYNC'
)

register(
    id='rma',
    entry_point='exarl.workflows.workflow_vault:RMA_ASYNC'
)

register(
    id='mlrma',
    entry_point='exarl.workflows.workflow_vault:ML_RMA'
)
