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
import numpy as np
import tensorflow
# import torch
import exarl
from exarl.utils.profile import ProfileConstants
import os
import exarl.utils.analyze_reward as ar
# from exarl.utils.renderDMC import render_tmp

exa_learner = exarl.ExaLearner()

# MPI communicator
comm = exarl.ExaComm.global_comm
rank = comm.rank
size = comm.size

# Run the learner, measure time
start = time.time()
exa_learner.run()
elapse = time.time() - start

# Print duration
if ProfileConstants.introspected():
    print("Rank", rank, "Time = ", elapse)
else:
    max_elapse = comm.reduce(np.float64(elapse), max, 0)
    elapse = comm.reduce(np.float64(elapse), sum, 0)
    if rank == 0:
        print("Average elapsed time = ", elapse / size)
        print("Maximum elapsed time = ", max_elapse)

print("Final number of episodes =", exa_learner.final_number_of_episodes())
print("Total reward =", exa_learner.final_total_reward())
print("Final rolling reward =", exa_learner.final_rolling_reward()[0])

# Save rewards vs. episodes plot
if rank == 0:
    ar.save_reward_plot()
