#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include "gs.hpp"

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  setupAide options;

  if(argc < 6) {
    if(rank == 0) printf(
        "usage: ./gs N nelX nelY nelZ SERIAL|CUDA|HIP|OPENCL <ogs_mode> <nRepetitions> <enable timers> <run dummy kernel> <use FP32> <GPU aware MPI> <DEVICE-ID>\n");

    MPI_Finalize();
    exit(1);
  }

  std::string threadModel;
  threadModel.assign(strdup(argv[5]));
  options.setArgs("THREAD MODEL", threadModel);

  options.setArgs("N", argv[1]);
  options.setArgs("NX", argv[2]);
  options.setArgs("NY", argv[3]);
  options.setArgs("NZ", argv[4]);

  options.setArgs("OGS MODE", ((argc > 6 && std::stoi(argv[6]) > 0) ? argv[6] : "0"));

  options.setArgs("NTESTS", (argc > 7 ? argv[7] : "100"));
  options.setArgs("ENABLED TIMER", (argc > 8 ? argv[8] : "0"));
  options.setArgs("DUMMY KERNEL", (argc > 9 ? argv[9] : "0"));
  options.setArgs("FLOAT TYPE", ((argc > 10 && std::stoi(argv[10])) ? "float" : "double"));
  options.setArgs("GPUMPI", ((argc > 11 && std::stoi(argv[11])) ? "1" : "0"));

  options.setArgs("DEVICE NUMBER", "LOCAL-RANK");
  if(argc > 12) {
    std::string deviceNumber;
    deviceNumber.assign(strdup(argv[12]));
    options.setArgs("DEVICE NUMBER", deviceNumber);
  }

  options.setArgs("DRIVER MODUS", "FALSE");

  gs(options);

  MPI_Finalize();
  return 0;
}
