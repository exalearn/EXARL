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
from mpi4py import MPI
import mpi4py.rc
mpi4py.rc.threads = False


def init(comm, learner_procs, procs_per_env):
    # global communicator
    global global_comm
    global_comm = comm
    global_rank = global_comm.rank
    global num_learners
    num_learners = learner_procs

    # Learner communicator
    global learner_comm
    learner_color = MPI.UNDEFINED
    if global_rank < num_learners:
        learner_color = 0
    learner_comm = global_comm.Split(learner_color, global_rank)

    # Agent communicator
    global agent_comm
    agent_color = MPI.UNDEFINED
    if (global_rank < num_learners) or ((global_rank + procs_per_env - 1) % procs_per_env == 0):
        agent_color = 0
    agent_comm = global_comm.Split(agent_color, global_rank)

    # Environment communicator
    if global_rank < num_learners:
        env_color = 0
    else:
        env_color = (int((global_rank - num_learners) / procs_per_env)) + 1
    global env_comm
    env_comm = global_comm.Split(env_color, global_rank)

# Function to test if a process is a learner
def is_learner():
    try:
        if agent_comm.rank < num_learners:
            return True
    except:
        return False

# Function to test if a process is an actor
def is_actor():
    try:
        if agent_comm.rank >= num_learners:
            return True
    except:
        return False

# Function to test if a process is an agent
def is_agent():
    if is_learner() or is_actor():
        return True
    else:
        return False
