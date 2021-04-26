from mpi4py import MPI
import ctypes
import os

_libdir = os.path.dirname(__file__)

if MPI._sizeof(MPI.Comm) == ctypes.sizeof(ctypes.c_int):
    MPI_Comm = ctypes.c_int
else:
    MPI_Comm = ctypes.c_void_p
_lib = ctypes.CDLL(os.path.join(_libdir, "libcomputePI.so"))
_lib.compute_pi.restype = ctypes.c_double
_lib.compute_pi.argtypes = [ctypes.c_int, MPI_Comm]


def compute_pi(N, comm):
    comm_ptr = MPI._addressof(comm)
    comm_val = MPI_Comm.from_address(comm_ptr)
    myPI = _lib.compute_pi(ctypes.c_int(N), comm_val)
    return myPI
