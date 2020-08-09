#include <mpi.h>
#include <occa.hpp>
#include "driver.hpp"

int main(int argc, char** argv)
{
  // start up MPI
  MPI_Init(&argc, &argv);

  if(argc != 2) {
    printf("usage: ./driver setupfile\n");

    MPI_Finalize();
    exit(1);
  }

  driver(argv[1], MPI_COMM_WORLD);

  MPI_Finalize();
  return 0;
}
