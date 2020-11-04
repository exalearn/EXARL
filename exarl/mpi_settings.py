import mpi4py.rc; mpi4py.rc.threads = False
from mpi4py import MPI

def init(procs_per_env):
    # World communicator
    global world_comm
    world_comm = MPI.COMM_WORLD
    world_rank = world_comm.rank

    # Agent communicator
    global agent_comm, agent_color
    agent_color = MPI.UNDEFINED
    if (world_rank == 0) or ((world_rank + procs_per_env - 1) % procs_per_env == 0):
        agent_color = 0
    agent_comm = world_comm.Split(agent_color, world_rank)

    # Environment communicator
    if world_rank == 0:
        env_color = 0
    else:
        env_color = (int((world_rank - 1) / procs_per_env)) + 1
    global env_comm
    env_comm = world_comm.Split(env_color, world_rank)

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