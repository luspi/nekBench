#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include "omp.h"
#include <unistd.h>
#include "mpi.h"
#include "occa.hpp"
#include "timer.hpp"
#include "setCompilerFlags.hpp"
#include "setupAide.hpp"
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
  bw(options);

  MPI_Finalize();
  exit(0);
}
