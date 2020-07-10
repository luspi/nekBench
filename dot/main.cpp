#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "omp.h"
#include <unistd.h>
#include "mpi.h"
#include "occa.hpp"
#include "meshBasis.hpp"
#include "kernelHelper.cpp"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cublas_v2.h>

occa::memory o_tmp;
occa::memory o_tmp2;
dfloat *tmp;
static occa::kernel kernel;

dfloat *drandAlloc(int N){

  dfloat *v = (dfloat*) calloc(N, sizeof(dfloat));

  for(int n=0;n<N;++n){
    v[n] = drand48();
  }

  return v;
}

dfloat weightedInnerProduct(dlong N, dlong Ncutoff, int Nblock, occa::memory &o_w, 
                            occa::memory &o_a, occa::memory &o_b, int global){

  dfloat globalwab = 0;
  kernel(N, o_w, o_a, o_b, o_tmp);

//  if(Nblock>Ncutoff){ /* add a second sweep if Nblock>Ncutoff */
//    sumKernel(Nblock, o_tmp, o_tmp2);
//    o_tmp2.copyTo(tmp);
//  }
//  else{
//    o_tmp.copyTo(tmp);
//  }

  o_tmp.copyTo(tmp);
  dfloat wab = 0;
  for(dlong n=0;n<Nblock;++n) wab += tmp[n];

  if(global)
      MPI_Allreduce(&wab, &globalwab, 1, MPI_DFLOAT, MPI_SUM, MPI_COMM_WORLD);
  else
      return wab;

  return globalwab;
}

double cloopInnerProduct(int N, double *a, double *b, double *w) {

  double result = 0;
  for(int i = 0; i < N; ++i)
    result += w[i]*a[i]*b[i];

  return result;

}

int main(int argc, char **argv){

  if(argc<5){
    printf("Usage: ./dot N numElements [NATIVE|OKL]+SERIAL|CUDA|OPENCL CPU|VOLTA [nRepetitions] [deviceId] [blockSize] [MPI]\n");
    return 1;
  }

  int rank = 0, size = 1;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int N = atoi(argv[1]);
  const dlong Nelements = atoi(argv[2]);
  std::string threadModel;
  threadModel.assign(strdup(argv[3]));

  std::string arch("");
  if(argc>=5)
    arch.assign(argv[4]);

  int Ntests = 100;
  if(argc>=6)
    Ntests = atoi(argv[5]);

  int deviceId = 0;
  if(argc>=7)
    deviceId = atoi(argv[6]);

  int blockSize = 256;
  if(argc>=8)
    blockSize = atoi(argv[7]);

  int global = 0;
  if(argc>=9)
    global = atoi(argv[8]);

  const int platformId = 0;

  const int Nq = N+1;
  const int Np = Nq*Nq*Nq;

  const dlong offset = Nelements*Np;

  // build device
  occa::device deviceSerial;
  occa::device deviceCuda;
  char deviceConfigSerial[BUFSIZ];
  char deviceConfigCuda[BUFSIZ];

  omp_set_num_threads(1);
  sprintf(deviceConfigSerial, "mode: 'Serial'");
  sprintf(deviceConfigCuda, "mode: 'CUDA', device_id: %d",deviceId);

  int Nthreads =  omp_get_max_threads();

  std::string deviceConfigStringSerial(deviceConfigSerial);
  std::string deviceConfigStringCuda(deviceConfigCuda);

  deviceSerial.setup(deviceConfigStringSerial);
  deviceCuda.setup(deviceConfigStringCuda);

  occa::env::OCCA_MEM_BYTE_ALIGN = USE_OCCA_MEM_BYTE_ALIGN;

  // populate device arrays
  dfloat *a = drandAlloc(Np*Nelements);
  dfloat *b = drandAlloc(Np*Nelements);
  dfloat *c = drandAlloc(Np*Nelements);


  for(int type = 0; type < 4; ++type) {

    int allresult[Ntests]{};
    double elapsedTime;

    // occa kernel (Serial)
    if(type == 0) {

      if(rank==0) {
        std::cout << "word size: " << sizeof(dfloat) << " bytes\n";
        std::cout << "active occa mode: " << deviceSerial.mode() << "\n";
      }

      // load kernel
      std::string kernelName = "weightedInnerProduct2";
      const int kernelVersion = 0; // hardwired for now
      kernelName += "_v" + std::to_string(kernelVersion);
      kernel = loadKernel(deviceSerial, threadModel, arch, kernelName, N, Nelements, blockSize);

      occa::memory o_a = deviceSerial.malloc(Np*Nelements*sizeof(dfloat), a);
      occa::memory o_b = deviceSerial.malloc(Np*Nelements*sizeof(dfloat), a);
      occa::memory o_c = deviceSerial.malloc(Np*Nelements*sizeof(dfloat), a);

      int Nblock  = ((Nelements*Np)+blockSize-1)/blockSize;
      if(Nblock < 1) Nblock = 1;
      if(rank == 0) std::cout << "blockSize: " << Nblock << "\n";

      tmp = drandAlloc(Nblock);
      o_tmp = deviceSerial.malloc(Nblock*sizeof(dfloat), tmp);
      o_tmp2 = deviceSerial.malloc(Nblock*sizeof(dfloat), tmp);

      // run kernel
      weightedInnerProduct(Nelements*Np, 0, Nblock, o_c, o_a, o_b, global);
      deviceSerial.finish();

      MPI_Barrier(MPI_COMM_WORLD);
      const double start = MPI_Wtime();

      for(int test=0;test<Ntests;++test) {
          allresult[test] = weightedInnerProduct(Nelements*Np, 0, Nblock, o_c, o_a, o_b, global);
      }

      MPI_Barrier(MPI_COMM_WORLD);
      elapsedTime = (MPI_Wtime() - start)/Ntests;
      deviceSerial.finish();

    // occa kernel (CUDA)
    } else  if(type == 1) {

      if(rank==0) {
        std::cout << "word size: " << sizeof(dfloat) << " bytes\n";
        std::cout << "active occa mode: " << deviceCuda.mode() << "\n";
      }

      // load kernel
      std::string kernelName = "weightedInnerProduct2";
      const int kernelVersion = 0; // hardwired for now
      kernelName += "_v" + std::to_string(kernelVersion);
      kernel = loadKernel(deviceCuda, threadModel, arch, kernelName, N, Nelements, blockSize);

      occa::memory o_a = deviceCuda.malloc(Np*Nelements*sizeof(dfloat), a);
      occa::memory o_b = deviceCuda.malloc(Np*Nelements*sizeof(dfloat), a);
      occa::memory o_c = deviceCuda.malloc(Np*Nelements*sizeof(dfloat), a);

      int Nblock  = ((Nelements*Np)+blockSize-1)/blockSize;
      if(Nblock < 1) Nblock = 1;
      if(rank == 0) std::cout << "blockSize: " << Nblock << "\n";

      tmp = drandAlloc(Nblock);
      o_tmp = deviceCuda.malloc(Nblock*sizeof(dfloat), tmp);
      o_tmp2 = deviceCuda.malloc(Nblock*sizeof(dfloat), tmp);

      // run kernel
      weightedInnerProduct(Nelements*Np, 0, Nblock, o_c, o_a, o_b, global);
      deviceCuda.finish();

      MPI_Barrier(MPI_COMM_WORLD);
      const double start = MPI_Wtime();

      for(int test=0;test<Ntests;++test) {
          allresult[test] = weightedInnerProduct(Nelements*Np, 0, Nblock, o_c, o_a, o_b, global);
      }

      MPI_Barrier(MPI_COMM_WORLD);
      elapsedTime = (MPI_Wtime() - start)/Ntests;
      deviceCuda.finish();

    // cuBlas
    } else if(type == 2) {

      occa::memory o_a = deviceCuda.malloc(Np*Nelements*sizeof(dfloat), a);
      occa::memory o_b = deviceCuda.malloc(Np*Nelements*sizeof(dfloat), a);
      occa::memory o_c = deviceCuda.malloc(Np*Nelements*sizeof(dfloat), a);

      if(rank==0) {
        std::cout << "word size: " << sizeof(dfloat) << " bytes\n";
        std::cout << "mode: cuBlas" << std::endl;
      }

      cublasHandle_t cublas;

      cublasStatus_t stat = cublasCreate(&cublas);
      if(stat != CUBLAS_STATUS_SUCCESS)
          printf("CUBLAS initialization failed!\n");

      MPI_Barrier(MPI_COMM_WORLD);
      const double start = MPI_Wtime();

      for(int test=0;test<Ntests;++test) {
        double result = 0;

        double alpha = 1.0;
        double beta = 0.0;
        cublasDsbmv(cublas, CUBLAS_FILL_MODE_LOWER, size, 0, &alpha, (double*)o_c.ptr(), 1, (double*)o_a.ptr(), 1, &beta, (double*)o_a.ptr(), 1);
        cublasDdot(cublas, Nelements*Np, (double*)o_a.ptr(), 1, (double*)o_b.ptr(), 1, &result);
        allresult[test] = result;
      }

      MPI_Barrier(MPI_COMM_WORLD);
      elapsedTime = (MPI_Wtime() - start)/Ntests;
      deviceCuda.finish();

    // C loop
    } else if(type == 3) {

      occa::memory o_a = deviceSerial.malloc(Np*Nelements*sizeof(dfloat), a);
      occa::memory o_b = deviceSerial.malloc(Np*Nelements*sizeof(dfloat), a);
      occa::memory o_c = deviceSerial.malloc(Np*Nelements*sizeof(dfloat), a);

      if(rank==0) {
        std::cout << "word size: " << sizeof(dfloat) << " bytes\n";
        std::cout << "mode: C loop" << std::endl;
      }

      MPI_Barrier(MPI_COMM_WORLD);
      const double start = MPI_Wtime();

      for(int test=0;test<Ntests;++test) {
        allresult[test] = cloopInnerProduct(Nelements*Np, (double*)o_a.ptr(), (double*)o_b.ptr(), (double*)o_c.ptr());
      }

      MPI_Barrier(MPI_COMM_WORLD);
      elapsedTime = (MPI_Wtime() - start)/Ntests;
      deviceSerial.finish();

    }

    double allsum = 0;
    for(int i = 0; i < Ntests; ++i)
        allsum += allresult[i];
    std::cout << "sum of all results = " << allsum << std::endl;

    // print statistics
    double GDOFPerSecond = size*(Nelements*N*N*N)/elapsedTime/1.e9;
    const long long bytesMoved = 3*Np;
    const double bw = (size*bytesMoved*Nelements/elapsedTime)/1.e9;
    //  double flopCount = ?;
    //  double gflops = (size*flopCount*Nelements/elapsed)/1.e9;
    if(rank==0) {
      std::cout << "MPItasks=" << size
                << " OMPthreads=" << Nthreads
                << " NRepetitions=" << Ntests
                << " N=" << N
                << " Nelements=" << size*Nelements
                << " blockSize=" << blockSize
                << " elapsed time=" << elapsedTime
                << " GDOF/s=" << GDOFPerSecond
                << " GB/s=" << bw
                << "\n\n";
    }

  }

  MPI_Finalize();
  exit(0);
}
