#include <iostream>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <sstream>
#include "omp.h"
#include <unistd.h>
#include "mpi.h"
#include "occa.hpp"
#include "timer.hpp"
#include "setupAide.hpp"
#include "setCompilerFlags.hpp"

void allred(setupAide &options, MPI_Comm mpiComm) {

  int mpiRank, mpiSize;
  MPI_Comm_size(mpiComm, &mpiSize);
  MPI_Comm_rank(mpiComm, &mpiRank);

  const int deviceId = 0;
  const int platformId = 0;

  bool driverModus = options.compareArgs("DRIVER MODUS", "TRUE");

  // build device
  occa::device device;
  char deviceConfig[BUFSIZ];
  
  std::string threadModel = options.getArgs("THREAD MODEL");

  if(strstr(threadModel.c_str(), "CUDA")) {
    sprintf(deviceConfig, "mode: 'CUDA', device_id: %d",deviceId);
  }else if(strstr(threadModel.c_str(),  "HIP")) {
    sprintf(deviceConfig, "mode: 'HIP', device_id: %d",deviceId);
  }else if(strstr(threadModel.c_str(),  "OPENCL")) {
    sprintf(deviceConfig, "mode: 'OpenCL', device_id: %d, platform_id: %d", deviceId, platformId);
  }else if(strstr(threadModel.c_str(),  "OPENMP")) {
    sprintf(deviceConfig, "mode: 'OpenMP' ");
  }else {
    sprintf(deviceConfig, "mode: 'Serial' ");
    omp_set_num_threads(1);
  }

  FILE *outputFile;

  if(driverModus && mpiRank == 0) {

    std::stringstream fname;
    fname << "allreduce_" << threadModel << "_ranks_" << mpiSize << ".txt";
    outputFile = fopen(fname.str().c_str(), "w");
    fprintf(outputFile, "%-10s %-10s %-10s\n", "words", "loops", "timing");

    std::cout << "allred: writing results to " << fname.str() << std::endl;

  }

  std::string deviceConfigString(deviceConfig);
  device.setup(deviceConfigString);
  occa::env::OCCA_MEM_BYTE_ALIGN = USE_OCCA_MEM_BYTE_ALIGN;

  std::stringstream outMode;
  outMode << "active occa mode: " << device.mode() << "\n\n";
  if(driverModus) {
    fprintf(outputFile, outMode.str().c_str());
    fflush(outputFile);
  } else {
    printf(outMode.str().c_str());
    fflush(stdout);
  }

  timer::init(mpiComm, device, 0);

  int max_message_size = 1e6;
  int Ntests = 100;

  for(int size = 1; size < max_message_size; size = (((int)(size*1.05)>size) ? (int)(size*1.05) : size+1)) {

    occa::memory o_mem_send = device.malloc(size * sizeof(double));
    occa::memory o_mem_recv = device.malloc(size * sizeof(double));
    device.finish();
    MPI_Barrier(mpiComm);
    timer::reset("allreduce");
    timer::tic("allreduce");

    for(int test = 0; test < Ntests; ++test)
      MPI_Allreduce(o_mem_send.ptr(), o_mem_recv.ptr(), size, MPI_DOUBLE, MPI_SUM, mpiComm);

    device.finish();
    MPI_Barrier(mpiComm);
    timer::toc("allreduce");
    timer::update();

    if(mpiRank == 0) {
      double elapsed = timer::query("allreduce", "HOST:MAX") / Ntests;
      if(driverModus)
        fprintf(outputFile, "%-10d %-10d %-10f\n", size, Ntests, elapsed/(double)Ntests);
      else
        printf("%-10d %-10d %-10f\n", size, Ntests, elapsed/(double)Ntests);
    }


  }

  if(driverModus)
    fclose(outputFile);

}
