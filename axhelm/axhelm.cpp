#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "omp.h"
#include <unistd.h>
#include "mpi.h"
#include "occa.hpp"
#include "meshBasis.hpp"
#include "setupAide.hpp"
namespace dA {
#include "drandalloc.hpp"
}

#include "kernelHelper.cpp"
#include "axhelmReference.cpp"

static occa::kernel axKernel;

std::string axhelmFormatStringForFilename(std::string in) {
  std::string out = in;
  size_t pos = out.find(" ");
  while(pos != std::string::npos) {
    out.replace(pos, 1, "");
    pos = out.find(" ", pos);
  }
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c){ return std::tolower(c); });
  return out;
}

void axhelm(setupAide &options, std::vector<std::string> optionsForFilename, MPI_Comm mpiComm) {

  const int N = std::stoi(options.getArgs("N"));
  const int Ndim = std::stoi(options.getArgs("NDIM"));
  const dlong Nelements = std::stol(options.getArgs("NELEMENTS"));
  std::string threadModel = options.getArgs("THREAD MODEL");
  std::string arch = options.getArgs("ARCH");
  int BKmode = std::stoi(options.getArgs("BKMODE"));
  int Ntests = std::stoi(options.getArgs("NTESTS"));
  int kernelVersion = std::stoi(options.getArgs("KERNELVERSION"));
  bool driverModus = options.compareArgs("DRIVER MODUS", "TRUE");

  int rank = 0, size = 1;
  MPI_Comm_rank(mpiComm, &rank);
  MPI_Comm_size(mpiComm, &size);

  const int deviceId = 0;
  const int platformId = 0;

  const int Nq = N + 1;
  const int Np = Nq * Nq * Nq;

  const dlong offset = Nelements * Np;

  const int assembled = 0;

  // build element nodes and operators
  dfloat* rV, * wV, * DrV;
  meshJacobiGQ(0,0,N, &rV, &wV);
  meshDmatrix1D(N, Nq, rV, &DrV);

  // build device
  occa::device device;
  char deviceConfig[BUFSIZ];

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

  int Nthreads =  omp_get_max_threads();
  std::string deviceConfigString(deviceConfig);
  device.setup(deviceConfigString);
  occa::env::OCCA_MEM_BYTE_ALIGN = USE_OCCA_MEM_BYTE_ALIGN;

  FILE *outputFile;
  if(rank == 0) {

    std::stringstream out;
    out << "word size: " << sizeof(dfloat) << " bytes\n";
    out << "active occa mode: " << device.mode() << "\n";
    if(BKmode) out << "BK mode enabled\n";

    if(driverModus) {

      std::stringstream fname;

      if(optionsForFilename.size() == 0)
        fname << "axhelm_" << threadModel << "_" << arch << "_N_" << N << "_ndim_" << Ndim << "_elements_" << Nelements << "_bkmode_" << BKmode << ".txt";
      else {
        fname << "axhelm";
        for(int i = 0; i < optionsForFilename.size(); ++i)
          fname << "_" << axhelmFormatStringForFilename(optionsForFilename[i]) << "_" << options.getArgs(optionsForFilename[i]);
        fname << ".txt";
      }

      outputFile = fopen(fname.str().c_str(), "w");
      fprintf(outputFile, out.str().c_str());

      std::cout << "axhelm: writing results to " << fname.str() << std::endl;

    } else
      std::cout << out.str();
  }

  // load kernel
  std::string kernelName = "axhelm";
  if(assembled) kernelName = "axhelm_partial";
  if(BKmode) kernelName = "axhelm_bk";
  if(Ndim > 1) kernelName += "_n" + std::to_string(Ndim);
  kernelName += "_v" + std::to_string(kernelVersion);
  axKernel = loadAxKernel(device, threadModel, arch, kernelName, N, Nelements, mpiComm);

  // populate device arrays
  dfloat* ggeo = dA::drandAlloc(Np * Nelements * p_Nggeo);
  dfloat* q    = dA::drandAlloc((Ndim * Np) * Nelements);
  dfloat* Aq   = dA::drandAlloc((Ndim * Np) * Nelements);

  occa::memory o_ggeo   = device.malloc(Np * Nelements * p_Nggeo * sizeof(dfloat), ggeo);
  occa::memory o_q      = device.malloc((Ndim * Np) * Nelements * sizeof(dfloat), q);
  occa::memory o_Aq     = device.malloc((Ndim * Np) * Nelements * sizeof(dfloat), Aq);
  occa::memory o_DrV    = device.malloc(Nq * Nq * sizeof(dfloat), DrV);

  dfloat lambda1 = 1.1;
  if(BKmode) lambda1 = 0;
  dfloat* lambda = (dfloat*) calloc(2 * offset, sizeof(dfloat));
  for(int i = 0; i < offset; i++) {
    lambda[i]        = 1.0; // don't change
    lambda[i + offset] = lambda1;
  }
  occa::memory o_lambda = device.malloc(2 * offset * sizeof(dfloat), lambda);

  // run kernel
  axKernel(Nelements, offset, o_ggeo, o_DrV, o_lambda, o_q, o_Aq);
  device.finish();
  MPI_Barrier(mpiComm);
  const double start = MPI_Wtime();

  for(int test = 0; test < Ntests; ++test)
    axKernel(Nelements, offset, o_ggeo, o_DrV, o_lambda, o_q, o_Aq);

  device.finish();
  MPI_Barrier(mpiComm);
  const double elapsed = (MPI_Wtime() - start) / Ntests;

  // check for correctness
  for(int n = 0; n < Ndim; ++n) {
    dfloat* x = q + n * offset;
    dfloat* Ax = Aq + n * offset;
    axhelmReference(Nq, Nelements, lambda1, ggeo, DrV, x, Ax);
  }
  o_Aq.copyTo(q);
  dfloat maxDiff = 0;
  for(int n = 0; n < Ndim * Np * Nelements; ++n) {
    dfloat diff = fabs(q[n] - Aq[n]);
    maxDiff = (maxDiff < diff) ? diff:maxDiff;
  }
  MPI_Allreduce(MPI_IN_PLACE, &maxDiff, 1, MPI_DFLOAT, MPI_SUM, mpiComm);
  if (rank == 0) {
    std::stringstream out;
    out << "Correctness check: maxError = " << maxDiff << "\n";
    if(driverModus)
      fprintf(outputFile, out.str().c_str());
    else
      std::cout << out.str();
  }

  // print statistics
  const dfloat GDOFPerSecond = (size * Ndim * (N * N * N) * Nelements / elapsed) / 1.e9;
  long long bytesMoved = (Ndim * 2 * Np + 7 * Np) * sizeof(dfloat); // x, Ax, geo
  if(!BKmode) bytesMoved += 2 * Np * sizeof(dfloat);
  const double bw = (size * bytesMoved * Nelements / elapsed) / 1.e9;
  double flopCount = Np * 12 * Nq;
  flopCount += 15 * Np;
  if(!BKmode) flopCount += 5 * Np;
  flopCount *= Ndim;
  double gflops = (size * flopCount * Nelements / elapsed) / 1.e9;
  if(rank == 0) {
    std::stringstream out;
    out << "MPItasks=" << size
        << " OMPthreads=" << Nthreads
        << " NRepetitions=" << Ntests
        << " Ndim=" << Ndim
        << " N=" << N
        << " Nelements=" << size * Nelements
        << " elapsed time=" << elapsed
        << " GDOF/s=" << GDOFPerSecond
        << " GB/s=" << bw
        << " GFLOPS/s=" << gflops
        << "\n";
    if(driverModus) {
      fprintf(outputFile, out.str().c_str());
      fclose(outputFile);
    } else
      std::cout << out.str();
  }

}
