import introbind as ib
import exarl as erl
import utils.analyze_reward as ar
import time
from utils.candleDriver import initialize_parameters

# Get run parameters using CANDLE
run_params = initialize_parameters()

# Create learner object and run
exa_learner = erl.ExaLearner(run_params)

# MPI communicator
comm = erl.ExaComm.global_comm
rank = comm.rank
size = comm.size

# Run the learner, measure time
ib.start()
start = time.time()
exa_learner.run()
elapse = time.time() - start
ib.stop()

# Compute and print average time
max_elapse = comm.reduce(elapse, max, 0)
elapse = comm.reduce(elapse, sum, 0)

if rank == 0:
    print("Average elapsed time = ", elapse / size)
    print("Maximum elapsed time = ", max_elapse)
    # Save rewards vs. episodes plot
    ar.save_reward_plot(run_params['output_dir'] + '/')
