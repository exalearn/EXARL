EXARL Environments
==================

Creating Custom Environments
----------------------------
- ExaRL uses OpenAI gym environments
- The ExaEnv class in ``ExaRL/exarl/env_base.py`` inherits from OpenAI GYM Wrapper class for including added functionality.
- Environments inherit from gym.Env

Example:-

.. code-block:: python

   class envName(gym.Env):
      ...

Register the environment in ``ExaRl/envs/__init__.py``

.. code-block:: python

   from gym.envs.registration import register

   register(
      id='fooEnv-v0',
      entry_point='envs.env_vault:FooEnv',
   )

The ID variable will be passed to exarl.make() to call the environment.

The file ``ExaRL/env/env_vault/__init__.py`` should include:

.. code-block:: python

   from envs.env_vault.foo_env import FooEnv

where ``ExaRL/envs/env_vault/foo_env.py`` is the file containing your envirnoment

Using Environment Written in a Lower Level Language
---------------------------------------------------
The following example illustrates using the C function of computing the value of PI in EXARL.

**computePI.h:**

.. code-block:: C

   #define MPICH_SKIP_MPICXX 1
   #define OMPI_SKIP_MPICXX 1
   #include <mpi.h>
   #include <stdio.h>

   #ifdef __cplusplus
   extern "C" {
   #endif
      extern void compute_pi(int, MPI_Comm);
   #ifdef __cplusplus
   }
   #endif


**computePI.c:**

.. code-block:: C

   #include <stdio.h>
   #include <mpi.h>

   double compute_pi(int N, MPI_Comm new_comm)
   {
      int rank, size;
      MPI_Comm_rank(new_comm, &rank);
      MPI_Comm_size(new_comm, &size);

      double h, s, x;
      h = 1.0 / (double) N;
      s = 0.0;
      for(int i=rank; i<N; i+=size)
      {
         x = h * ((double)i + 0.5);
         s += 4.0 / (1.0 + x*x);
      }
      return (s * h);
   }

Compile the C/C++ code and create a shared object (*.so file).
Create a python wrapper (Ctypes wrapper is shown).

**computePI.py:**

.. code-block:: python

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

In your environment code, just import the function and use it regularly.

**test_computePI.py:**

.. code-block:: python

   from mpi4py import MPI
   import numpy as np
   import pdb
   import computePI as cp

   def main():
      comm = MPI.COMM_WORLD
      myrank = comm.Get_rank()
      nprocs = comm.Get_size()

      if myrank == 0:
         N = 100
      else:
         N = None

      N = comm.bcast(N, root=0)
      num = 4
      color = int(myrank/num)
      newcomm = comm.Split(color, myrank)

      mypi = cp.compute_pi(N, newcomm)
      pi = newcomm.reduce(mypi, op=MPI.SUM, root=0)

      newrank = newcomm.rank
      if newrank==0:
         print(pi)

   if __name__ == '__main__':
      main()
