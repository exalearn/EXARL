from mpi4py import MPI
import numpy as np
import struct
import sys
#################################################################
# Global constants
#################################################################
comm = MPI.COMM_WORLD
batch_size = 64
model_size = 208
#################################################################
# Functions
#################################################################
# MPI.LOCK_EXCLUSIVE
# MPI.LOCK_SHARED

def run_lock_test(buffer_size, iter=10000):

        buf = bytearray(buffer_size*[8])
        win_size = buffer_size if comm.rank == 0 else 0
        single_win = MPI.Win.Allocate(win_size, MPI.BYTE.Get_size(), comm=comm)

        if comm.rank == 0:
            single_win.Lock(0)
            single_win.Put(buf, target_rank=0)
            single_win.Unlock(0)
            comm.Barrier()
            comm.Barrier()
        else:
            comm.Barrier()
            recv_buf = bytearray(buffer_size)

            # exclusive lock
            start = MPI.Wtime()
            for _ in range(iter):
                single_win.Lock(0)
                single_win.Get(recv_buf, target_rank=0)
                single_win.Unlock(0)
            end = MPI.Wtime()

            exclusive_avg_time = ((end - start)* 1e6) / iter
            comm.Barrier()

            # shared lock
            start = MPI.Wtime()
            for _ in range(iter):
                single_win.Lock(0, lock_type=MPI.LOCK_SHARED)
                single_win.Get(recv_buf, target_rank=0)
                single_win.Unlock(0)
            end = MPI.Wtime()

            shared_avg_time = ((end - start)* 1e6) / iter
            if comm.rank == 1:
                print("{:>20} {:>20} {:>20} {:>20}".format(buffer_size, iter, round(shared_avg_time,2), round(exclusive_avg_time,2)))

        comm.Barrier()
        single_win.Free()
#################################################################
# Main
#################################################################
if __name__=="__main__":

    if len(sys.argv) != 3:
        print("Usage : {} <init_size> <generations>".format(sys.argv[0]))
        sys.exit(1)

    init_size=int(sys.argv[1])
    generations=int(sys.argv[2])

    if comm.rank == 1:
        print("# --------------------------------------------")
        print("# LOCK RMA TEST")
        print("# readers {}".format(comm.size - 1))
        print("# --------------------------------------------")
        print("{:>20} {:>20} {:>20} {:>20}".format("# Bytes","# Repetition","# t_shared[usec]", "# t_exclusive[usec]"))

    buff_size = init_size
    for _ in range(generations):
        run_lock_test(buff_size)
        buff_size*=2
