import exarl as erl
import driver.candleDriver as cd
import time

## Get run parameters using CANDLE
run_params = cd.initialize_parameters()

## Create learner object and run
exa_learner = erl.ExaLearner(run_params)
run_type = exa_learner.env.run_type
start = time.time()
exa_learner.run(run_type) # can be either 'static' or 'dynamic'
stop = time.time()
print("Elapsed time = ", stop - start)
