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
