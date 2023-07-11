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

exa_learner = exarl.ExaLearner()

# MPI communicator
comm = exarl.ExaComm.global_comm
rank = comm.rank
size = comm.size

print("Rank: {} Size: {} Learner: {} Agent: {} Env: {}".format(exarl.ExaComm.global_comm.rank, 
                                                               exarl.ExaComm.global_comm.size, 
                                                               exarl.ExaComm.learner_comm,
                                                               exarl.ExaComm.agent_comm,
                                                               exarl.ExaComm.env_comm), flush=True)
if exarl.ExaComm.learner_comm is not None:
    print("Rank: {} Learner: {} Size: {}".format(exarl.ExaComm.global_comm.rank,
                                                 exarl.ExaComm.learner_comm.rank,
                                                 exarl.ExaComm.learner_comm.size), flush=True)
comm.barrier()
if exarl.ExaComm.agent_comm is not None:
    print("Rank: {} Agent: {} Size: {}".format(exarl.ExaComm.global_comm.rank,
                                                 exarl.ExaComm.agent_comm.rank,
                                                 exarl.ExaComm.agent_comm.size), flush=True)
comm.barrier()
if exarl.ExaComm.env_comm is not None:
    print("Rank: {} Env: {} Size: {}".format(exarl.ExaComm.global_comm.rank,
                                                 exarl.ExaComm.env_comm.rank,
                                                 exarl.ExaComm.env_comm.size), flush=True)