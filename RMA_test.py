'''
int MPI_Win_create(void *base, MPI_Aint size, int disp_unit, MPI_Info info,
MPI_Comm comm, MPI_Win *win)

int MPI_Win_free(MPI_Win *win)

int MPI_Win_allocate(MPI Aint size, int disp unit, MPI_Info info,
MPI_Comm comm, void *baseptr, MPI_Win *win)

int MPI_Put(const void *origin_addr, int origin_count,
MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp,
int target_count, MPI_Datatype target_datatype, MPI_Win win)

int MPI_Get(void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
int target_rank, MPI_Aint target_disp, int target_count,
MPI_Datatype target_datatype, MPI_Win win)

int MPI_Win_lock(int lock type, int rank, int assert, MPI_Win win)

int MPI_Win_unlock(int rank, MPI_Win win)
'''

from mpi4py import MPI
import numpy as np
import pickle
import sys

comm = MPI.COMM_WORLD
rank = comm.rank

size = 10
disp = MPI.DOUBLE.Get_size()
if comm.rank == 0:
    buff = np.zeros(size, dtype='d')
else:
    buff = None

win = MPI.Win.Create(buff, disp, comm = comm)

if rank == 0:
    buff[5:10] = np.arange(5) + 5

comm.Barrier()

if rank > 0:
    holder = np.zeros(size)
    holder[:5] = np.arange(5)

    win.Lock(0)
    win.Put([holder, 5], 0, target=0)
    win.Unlock(0)

    win.Lock(0)
    win.Get(holder[5:10], 0, target=[5,3])
    win.Unlock(0)
    print('holder = ', holder)

comm.Barrier()

if rank == 0:
    print('buffer = ', buff)
