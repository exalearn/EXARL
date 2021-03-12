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

/*
void main()
{
        MPI_Init(NULL, NULL);
        int rank, size;
        MPI_Comm_size(MPI_COMM_WORLD,&size);
        MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        int n = 10;
        double mypi, pi;
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        mypi = compute_pi(n, MPI_COMM_WORLD);

        MPI_Reduce(&mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD);

        if(rank == 0)
                printf("PI = %f\n", pi);
        MPI_Finalize();
}
*/

