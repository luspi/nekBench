#include <stdlib.h>
#include <mpi.h>
#include "bw.hpp"

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  if(argc < 2) {
    printf("Usage: ./bw SERIAL|CUDA|OPENCL\n");
    return 1;
  }

  setupAide options;
  options.setArgs("THREAD MODEL", argv[1]);
  options.setArgs("DRIVER MODUS", "FALSE");
  bw(options);

  MPI_Finalize();
  exit(0);
}
