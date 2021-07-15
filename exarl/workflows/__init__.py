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
    id='mlrma',
    entry_point='exarl.workflows.workflow_vault:ML_RMA'
)

register(
    id='seed_v2',
    entry_point='exarl.workflows.workflow_vault:SEED_v2'
)




register(
    id='non_blocking_async_v2',
    entry_point='exarl.workflows.workflow_vault:NON_BLOCKING_ASYNC_v2'
)
register(
    id='mlrma_queue',
    entry_point='exarl.workflows.workflow_vault:ML_RMA_QUEUE'
)
register(
    id='mlrma_queue_short',
    entry_point='exarl.workflows.workflow_vault:ML_RMA_QUEUE_SHORT'
)
register(
    id='rma_queue_pop_all',
    entry_point='exarl.workflows.workflow_vault:RMA_QUEUE_POP_ALL'
)
register(
    id='seed',
    entry_point='exarl.workflows.workflow_vault:SEED'
)
register(
    id='seed_a2c',
    entry_point='exarl.workflows.workflow_vault:SEED_A2C'
)
register(    
    id='mlasync',
    entry_point='exarl.workflows.workflow_vault:ML_ASYNC'
)
