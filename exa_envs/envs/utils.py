import os
import sys
import shutil
import time
import datetime as dt

def print_status(msg, *args, comm_rank=None, showtime=True, barrier=True, allranks=False, flush=True):
    if comm_rank == None:
        comm_rank = 0
    #if not comm:
    #    comm = MPI.COMM_WORLD
    #if barrier:
    #    comm.Barrier()
    if showtime:
        s = dt.datetime.now().strftime('[%H:%M:%S.%f] ')
    else:
        s = ''
    if allranks:
        s += '{' + str(comm_rank) + '} '
    else:
        if comm_rank > 0:
            return
    s += msg.format(*args)
    print(s, flush=flush)
