#include <stdio.h>
#include <mpi.h>

int compute_pi(int N, MPI_Comm new_comm)
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
