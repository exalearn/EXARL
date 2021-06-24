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
    entry_point='exarl.workflows.workflow_vault:RMA'
)

register(
    id='tester',
    entry_point='exarl.workflows.workflow_vault:TESTER'
)

register(
    id='random',
    entry_point='exarl.workflows.workflow_vault:RANDOM'
    id='rma_v2',
    entry_point='exarl.workflows.workflow_vault:RMA_ASYNC_v2'
)
