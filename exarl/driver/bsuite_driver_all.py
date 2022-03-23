# This material was prepared as an account of work sponsored by an agency of the
# United States Government.  Neither the United States Government nor the United
# States Department of Energy, nor Battelle, nor any of their employees, nor any
# jurisdiction or organization that has cooperated in the development of these
# materials, makes any warranty, express or implied, or assumes any legal
# liability or responsibility for the accuracy, completeness, or usefulness or
# any information, apparatus, product, software, or process disclosed, or
# represents that its use would not infringe privately owned rights. Reference
# herein to any specific commercial product, process, or service by trade name,
# trademark, manufacturer, or otherwise does not necessarily constitute or imply
# its endorsement, recommendation, or favoring by the United States Government
# or any agency thereof, or Battelle Memorial Institute. The views and opinions
# of authors expressed herein do not necessarily state or reflect those of the
# United States Government or any agency thereof.
#                 PACIFIC NORTHWEST NATIONAL LABORATORY
#                            operated by
#                             BATTELLE
#                             for the
#                   UNITED STATES DEPARTMENT OF ENERGY
#                    under Contract DE-AC05-76RL01830
from numpy.random.mtrand import seed
import exarl as erl
# import exarl.utils.analyze_reward as ar
import time
from exarl.utils.candleDriver import lookup_params, run_params
from exarl.utils.introspect import *
import numpy as np

from bsuite import sweep
from tqdm import tqdm

# Experiment parameters. 
# Edit: 1. Where you want to start
#       2. What environments to exclude
#       3. The max seed number
start_id = 0
excluded_envs = ['cartpole_swingup',
                 'mountain_car',
                 'mountain_car_noise',
                 'mountain_car_scale']
max_seed_number = 1
# End of experiment parameters

for env_id in tqdm(sweep.SWEEP[start_id:]):

    # Only use seed number 0 until we can parallelize
    bsuite_id, seed_number = env_id.split('/')
    if int(seed_number) > max_seed_number or bsuite_id in excluded_envs:
        continue

    print("Current Env: ", env_id)
    run_params["bsuite_id"], run_params['seed_number'] = bsuite_id, seed_number
    run_params["n_episodes"] = sweep.EPISODES[env_id]

	# Create learner object and run
    exa_learner = erl.ExaLearner()

    # MPI communicator
    comm = erl.ExaComm.global_comm
    rank = comm.rank
    size = comm.size

    writeDir = lookup_params("introspector_dir")
    if writeDir is not None:
        ibLoadReplacement(comm, writeDir)

    # Run the learner, measure time
    ib.start()
    start = time.time()
    exa_learner.run()
    elapse = time.time() - start
    ib.stop()

    if ibLoaded():
        print("Rank", comm.rank, "Time = ", elapse)
    else:
        max_elapse = comm.reduce(np.float64(elapse), max, 0)
        elapse = comm.reduce(np.float64(elapse), sum, 0)
        if rank == 0:
            print("Average elapsed time = ", elapse / size)
            print("Maximum elapsed time = ", max_elapse)

    # if rank == 0:
    # 	# Save rewards vs. episodes plot
    # 	ar.save_reward_plot()

    ibWrite(writeDir)
