from mpi4py import MPI
import mpi4py.rc
mpi4py.rc.threads = False
import numpy as np
import functools

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

def checkNumpy(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not isinstance(args[0], np.ndarray):
            theArgs = list(args)
            theArgs[0] = np.asarray(theArgs[0])
            result = func(*theArgs, **kwargs)
        else:
            result = func(*args, **kwargs)
        return result
    return wrapper

@checkNumpy
def agent_send(*args, **kwargs):
    return agent_comm.send(*args, **kwargs)

@checkNumpy
def agent_bcast(*args, **kwargs):
    return agent_comm.bcast(*args, **kwargs)

@checkNumpy
def env_send(*args, **kwargs):
    return env_comm.send(*args, **kwargs)

@checkNumpy
def env_bcast(*args, **kwargs):
    return env_comm.bcast(*args, **kwargs)
