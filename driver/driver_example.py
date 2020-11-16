import mpi4py.rc; mpi4py.rc.threads = False; mpi4py.rc.recv_mprobe = False
import exarl as erl
from utils.candleDriver import initialize_parameters
import time
import utils.analyze_reward as ar
from mpi4py import MPI

# MPI communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

## Get run parameters using CANDLE
run_params = initialize_parameters()

## Create learner object and run
exa_learner = erl.ExaLearner(run_params)
learner_type = run_params['learner_type']

# Run the learner, measure time
start = time.time()
exa_learner.run(learner_type)
elapse = time.time() - start

# Compute and print average time
max_elapse = comm.reduce(elapse, op=MPI.MAX, root=0)
elapse = comm.reduce(elapse, op=MPI.SUM, root=0)

if rank == 0:
    print("Average elapsed time = ", elapse/size)
    print("Maximum elapsed time = ", max_elapse)
    # Save rewards vs. episodes plot
    ar.save_reward_plot(run_params['output_dir']+'/')
