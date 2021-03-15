from mpi4py import MPI
import mpi4py.rc
mpi4py.rc.threads = False


def init(comm, procs_per_env):
    # global communicator
    global global_comm
    global_comm = comm
    global_rank = global_comm.rank

    # Agent communicator
    global agent_comm
    agent_color = MPI.UNDEFINED
    if (global_rank == 0) or ((global_rank + procs_per_env - 1) % procs_per_env == 0):
        agent_color = 0
    agent_comm = global_comm.Split(agent_color, global_rank)

    # Environment communicator
    if global_rank == 0:
        env_color = 0
    else:
        env_color = (int((global_rank - 1) / procs_per_env)) + 1
    global env_comm
    env_comm = global_comm.Split(env_color, global_rank)

# Function to test if a process is a learner
def is_learner():
    try:
        if agent_comm.rank == 0:
            return True
    except:
        return False

# Function to test if a process is an actor
def is_actor():
    try:
        if agent_comm.rank > 0:
            return True
    except:
        return False

# Function to test if a process is an agent
def is_agent():
    if is_learner() or is_actor():
        return True
    else:
        return False
