#include <stdlib.h>
#include <mpi.h>
#include "axhelm.hpp"

int main(int argc, char** argv)
{
  if(argc < 6) {
    printf(
      "Usage: ./axhelm N Ndim numElements [NATIVE|OKL]+SERIAL|CUDA|HIP|OPENCL CPU|VOLTA [BKmode] [nRepetitions] [kernelVersion]\n");
    return 1;
  }

  MPI_Init(&argc, &argv);

  setupAide options;
  options.setArgs("N", argv[1]);
  options.setArgs("NDIM", argv[2]);
  options.setArgs("NELEMENTS", argv[3]);
  options.setArgs("THREAD MODEL", argv[4]);
  options.setArgs("ARCH", (argc > 5 ? argv[5] : ""));
  options.setArgs("BKMODE", (argc > 6 ? argv[6] : "1"));
  options.setArgs("NTESTS", (argc > 7 ? argv[7] : "100"));
  options.setArgs("KERNELVERSION", (argc > 8 ? argv[8] : "0"));
  options.setArgs("DRIVER MODUS", "FALSE");

  axhelm(options);

  MPI_Finalize();
  exit(0);
}
