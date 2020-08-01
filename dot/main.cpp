#include <mpi.h>
#include "dot.hpp"

int main(int argc, char** argv)
{

  MPI_Init(&argc, &argv);

  if(argc < 5) {
    printf(
      "Usage: ./dot N numElements [NATIVE|OKL]+SERIAL|CUDA|OPENCL CPU|VOLTA [nRepetitions] [deviceId] [blockSize] [MPI]\n");
    return 1;
  }

  setupAide options;
  options.setArgs("N", argv[1]);
  options.setArgs("N ELEMENTS", argv[2]);
  options.setArgs("THREAD MODEL", argv[3]);
  options.setArgs("ARCH", (argc >= 5 ? argv[4] : ""));
  options.setArgs("N TESTS", (argc >= 6 ? argv[5] : "100"));
  options.setArgs("DEVICE ID", (argc >= 7 ? argv[6] : "0"));
  options.setArgs("BLOCK SIZE", (argc >= 8 ? argv[7] : "256"));
  options.setArgs("GLOBAL", (argc >= 9 ? argv[8] : "0"));

  dot(options);

  MPI_Finalize();
  exit(0);
}
