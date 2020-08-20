#include <mpi.h>
#include <occa.hpp>
#include "parReader.hpp"
#include "driver.hpp"

int main(int argc, char** argv)
{
  // start up MPI
  MPI_Init(&argc, &argv);

  std::string parfile = "";
  if(argc > 1)
    parfile = argv[1];
  else
    std::cout << "Using default parameters." << std::endl;

  driver(parfile, MPI_COMM_WORLD);

  MPI_Finalize();
  return 0;
}
