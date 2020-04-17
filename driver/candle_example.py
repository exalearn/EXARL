import exarl as erl
import driver.candleDriver as cd
import time

## Get run parameters using CANDLE
run_params = cd.initialize_parameters()
results_dir = run_params['output_dir']+run_params['experiment_id']+'/'+run_params['run_id']

## Define agent and env
agent_id = 'agents:'+run_params['agent']
env_id   = 'envs:'+run_params['env']

## Create learner object and run
exa_learner = erl.ExaLearner(agent_id,env_id)
exa_learner.set_config(run_params)
exa_learner.set_results_dir(results_dir)
run_type = exa_learner.env.run_type
start = time.time()
exa_learner.run(run_type) # can be either 'static' or 'dynamic'
stop = time.time()
print("Elapsed time = ", stop - start)
