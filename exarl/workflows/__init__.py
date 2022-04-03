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
    entry_point='exarl.workflows.workflow_vault:RMA'
)

register(
    id='tester',
    entry_point='exarl.workflows.workflow_vault:TESTER'
)

register(
    id='random',
    entry_point='exarl.workflows.workflow_vault:RANDOM'
)

register(
    id='simple',
    entry_point='exarl.workflows.workflow_vault:SIMPLE'
)

register(
    id='simple_async',
    entry_point='exarl.workflows.workflow_vault:SIMPLE_ASYNC'
)

register(
    id='simple_rma',
    entry_point='exarl.workflows.workflow_vault:SIMPLE_RMA'
)
