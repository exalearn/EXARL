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
import time
import exarl
from exarl.utils.globals import ExaGlobals

import numpy as np
from tqdm import tqdm
from bsuite import sweep

"""
This is a driver that steps through all of Bsuite
with a single configuration.  It runs each environment
one after another.  There are three parameters that
can be adjusted from command line:

    - start_id - what environment index to start from
    - max_seed_number - max seeds to run per environment
    - max_episodes - max number of episodes to run per environment

To adjust these from command line, be sure to set the
environment to Bsuite-v0 as that is where candlelib will pick
up the arguments from.  If you don't then this driver will
try to run everything with full episodes.

We added max_episodes to differentiate the episodes for this
driver as the user probably will have a learner_cfg.json file
configured for regular experiments.  If they are sure they
want to limit the episodes they will have to add the --env
flag.  By setting max_episodes to -1, we will run the
number of episodes given by Bsuite sweep.
"""

# Experiment parameters.
# TODO: Fix these envs.
excluded_envs = ['cartpole_swingup',
                 'mountain_car',
                 'mountain_car_noise',
                 'mountain_car_scale']
start_id = ExaGlobals.lookup_params("driver_start_id")
max_seed_number = ExaGlobals.lookup_params("driver_max_seeds")
max_episodes = ExaGlobals.lookup_params("driver_max_episodes")
# End of experiment parameters

for env_id in tqdm(sweep.SWEEP[start_id:]):

    # Only use seed number 0 until we can parallelize
    bsuite_id, seed_number = env_id.split('/')
    if int(seed_number) > max_seed_number or bsuite_id in excluded_envs:
        continue

    episodes = sweep.EPISODES[env_id] if max_episodes == -1 else max_episodes
    ExaGlobals.set_param("n_episodes", episodes)
    ExaGlobals.set_param("bsuite_id", bsuite_id)
    ExaGlobals.set_param('seed_number', seed_number)

    print("Current Env:", ExaGlobals.lookup_params("bsuite_id"), "Seed:", ExaGlobals.lookup_params('seed_number'),
          "Episodes:", ExaGlobals.lookup_params("n_episodes"), "Steps:", ExaGlobals.lookup_params("n_steps"))

    # Create learner object and run
    exa_learner = exarl.ExaLearner()

    # MPI communicator
    comm = exarl.ExaComm.global_comm
    rank = comm.rank
    size = comm.size

    # Run the learner, measure time
    start = time.time()
    exa_learner.run()
    elapse = time.time() - start

    max_elapse = comm.reduce(np.float64(elapse), max, 0)
    elapse = comm.reduce(np.float64(elapse), sum, 0)
    if rank == 0:
        print("Average elapsed time = ", elapse / size)
        print("Maximum elapsed time = ", max_elapse)
