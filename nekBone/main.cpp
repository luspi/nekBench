#include <mpi.h>
#include "nekBone.hpp"

int main(int argc, char** argv)
{
  // start up MPI
  MPI_Init(&argc, &argv);

  if(argc != 2) {
    printf("usage: ./nekBone setupfile\n");

    MPI_Finalize();
    exit(1);
  }

  // if argv > 2 then should load input data from argv
  setupAide options(argv[1]);

  options.setArgs("DRIVER MODUS", "FALSE");

  nekBone(options, MPI_COMM_WORLD);

  MPI_Finalize();
  return 0;
}
