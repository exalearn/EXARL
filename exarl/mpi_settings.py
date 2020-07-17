import mpi4py.rc; mpi4py.rc.threads = False
from mpi4py import MPI

def init(env_procs):
    # World communicator
    world_comm = MPI.COMM_WORLD
    world_rank = world_comm.rank
    world_size = world_comm.size

    global env_ranks
    env_ranks = []
    for i in range(world_size):
        if i/env_procs==0:
            env_ranks.append(i)
    global env_color
    env_color = int(world_rank/(env_procs))
    global env_comm
    env_comm = MPI.COMM_WORLD.Split(env_color ,key=0)
    world_comm.barrier()

    # TODO: we need to make this properly done in the future 
    global agent_comm
    agent_comm = MPI.COMM_WORLD
