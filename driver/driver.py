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
import exarl as erl
import utils.analyze_reward as ar
import time
from utils.candleDriver import initialize_parameters
from utils.introspect import ib
from utils.introspect import ibLoaded
import numpy as np

# Create learner object and run
exa_learner = erl.ExaLearner()

# MPI communicator
comm = erl.ExaComm.global_comm
rank = comm.rank
size = comm.size


# Run the learner, measure time
ib.start()
start = time.time()
exa_learner.run()
elapse = time.time() - start
max_elapse = comm.reduce(np.float64(elapse), max, 0)
elapse = comm.reduce(np.float64(elapse), sum, 0)
ib.stop()

if rank == 0:
    print("Average elapsed time = ", elapse / size)
    print("Maximum elapsed time = ", max_elapse)
    # Save rewards vs. episodes plot
    ar.save_reward_plot()
