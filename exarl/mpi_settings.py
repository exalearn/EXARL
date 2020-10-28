import mpi4py.rc; mpi4py.rc.threads = False
from mpi4py import MPI

def init(env_procs):
    # World communicator
    world_comm = MPI.COMM_WORLD
    world_rank = world_comm.rank
    world_size = world_comm.size

    # TODO: we need to make this properly done in the future
    global agent_comm
    agent_comm = world_comm
    
    procs_per_env = 1
    env_color = int((world_rank+1)/procs_per_env)
    global env_comm
    env_comm = world_comm.Split(env_color, world_rank)

    world_comm.barrier()
