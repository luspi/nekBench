#include <stdlib.h>
#include <mpi.h>
#include "allred.hpp"

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  if(argc < 2) {
    printf("Usage: ./allred SERIAL|CUDA|OPENCL\n");
    return 1;
  }

  setupAide options;
  options.setArgs("THREAD MODEL", argv[1]);
  options.setArgs("DRIVER MODUS", "FALSE");
  allred(options, MPI_COMM_WORLD);

  MPI_Finalize();
  exit(0);
}
