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

void bw(setupAide &options) {
    
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

  if(driverModus) {

    std::stringstream fname;
    fname << "bw_" << threadModel << ".txt";
    outputFile = fopen(fname.str().c_str(), "w");

    std::cout << "bw: writing results to " << fname.str() << std::endl;

  }

  int Nthreads =  omp_get_max_threads();
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

  occa::properties props;
  props["defines"].asObject();
  props["includes"].asArray();
  props["header"].asArray();
  props["flags"].asObject();
  setCompilerFlags(device, props);
  occa::kernel triadKernel = device.buildKernel(DBP "kernel/triad.okl", "triad", props);

  timer::init(MPI_COMM_WORLD, device, 0);

  {
    const int Ntests = 1000;
    const int N[] = {1, 1000 * 512, 2000 * 512, 4000 * 512, 8000 * 512};
    const int Nsize = sizeof(N) / sizeof(int);
    const int nWords = N[sizeof(N) / sizeof(N[0]) - 1];
    occa::memory o_a = device.malloc(nWords * sizeof(double));
    occa::memory o_b = device.malloc(nWords * sizeof(double));
    occa::memory o_c = device.malloc(nWords * sizeof(double));

    for(int test = 0; test < Ntests; ++test) triadKernel(1000, 1.0, o_a, o_b, o_c);

    for(int i = 0; i < Nsize; ++i) {
      long long int bytes = 3 * N[i] * sizeof(double);
      device.finish();
      timer::reset("triad");
      timer::tic("triad");
      for(int test = 0; test < Ntests; ++test) triadKernel(N[i], 1.0, o_a, o_b, o_c);
      device.finish();
      timer::toc("triad");
      timer::update();
      double elapsed = timer::query("triad", "HOST:MAX") / Ntests;
      std::stringstream out;
      out << "triad stream: "
          << N[i] << " words, "
          << elapsed << " s, "
          << bytes / elapsed << " bytes/s\n";
      if(driverModus) {
        fprintf(outputFile, out.str().c_str());
        fflush(outputFile);
      } else {
        printf(out.str().c_str());
        fflush(stdout);
      }
    }
    o_a.free();
    o_b.free();
    o_c.free();

    occa::memory o_wrk = device.malloc(3 * nWords * sizeof(double));
    o_a = o_wrk + 0 * nWords * sizeof(double);
    o_b = o_wrk + 1 * nWords * sizeof(double);
    o_c = o_wrk + 2 * nWords * sizeof(double);
    for(int i = 0; i < Nsize; ++i) {
      long long int bytes = 3 * N[i] * sizeof(double);
      device.finish();
      timer::reset("triad");
      timer::tic("triad");
      for(int test = 0; test < Ntests; ++test) triadKernel(N[i], 1.0, o_a, o_b, o_c);
      device.finish();
      timer::toc("triad");
      timer::update();
      double elapsed = timer::query("triad", "HOST:MAX") / Ntests;
      std::stringstream out;
      out << "triad stream subBuffer: "
          << N[i] << " words, "
          << elapsed << " s, "
          << bytes / elapsed << " bytes/s\n";
      if(driverModus) {
        fprintf(outputFile, out.str().c_str());
        fflush(outputFile);
      } else {
        printf(out.str().c_str());
        fflush(stdout);
      }
    }
    o_wrk.free();
  }

  if(driverModus)
    fprintf(outputFile, "\n");
  else
    printf("\n");

  {
    const int N[] =
    {1, 4000, 8000, 50000, 100000, 150000, 300000, 2000 * 512, 4000 * 512, 8000 * 512};
    const int Nsize = sizeof(N) / sizeof(int);
    const int nWords = N[sizeof(N) / sizeof(N[0]) - 1];
    props["mapped"] = true;
    occa::memory h_u = device.malloc(nWords * sizeof(double), props);
    occa::memory o_a = device.malloc(nWords * sizeof(double));

    for(int i = 0; i < Nsize; ++i) {
      int Ntests = 5000;
      if(N[i] > 10000) Ntests = 100;
      const long long int bytes = N[i] * sizeof(double);
      void* ptr = h_u.ptr(props);
      device.finish();
      timer::reset("memcpyDH");
      timer::tic("memcpyDH");
      for(int test = 0; test < Ntests; ++test) o_a.copyTo(ptr, bytes);
      device.finish();
      timer::toc("memcpyDH");
      timer::update();
      const double elapsed = timer::query("memcpyDH", "HOST:MAX") / Ntests;
      std::stringstream out;
      out << "D->H memcpy " <<  N[i] << " words, "
          << elapsed << " s, "
          << bytes / elapsed << " bytes/s\n";
      if(driverModus) {
        fprintf(outputFile, out.str().c_str());
        fflush(outputFile);
      } else {
        printf(out.str().c_str());
        fflush(stdout);
      }
    }
    
    if(driverModus)
      fprintf(outputFile, "\n");
    else
      printf("\n");

    for(int i = 0; i < Nsize; ++i) {
      int Ntests = 5000;
      if(N[i] > 10000) Ntests = 100;
      const long long int bytes = N[i] * sizeof(double);
      void* ptr = h_u.ptr(props);
      device.finish();
      timer::reset("memcpyHD");
      timer::tic("memcpyHD");
      for(int test = 0; test < Ntests; ++test) o_a.copyFrom(ptr, bytes);
      device.finish();
      timer::toc("memcpyHD");
      timer::update();
      const double elapsed = timer::query("memcpyHD", "HOST:MAX") / Ntests;
      std::stringstream out;
      out << "H->D memcpy " <<  N[i] << " words, "
          << elapsed << " s, "
          << bytes / elapsed << " bytes/s\n";
      if(driverModus) {
        fprintf(outputFile, out.str().c_str());
        fflush(outputFile);
      } else {
        printf(out.str().c_str());
        fflush(stdout);
      }
    }

    h_u.free();
    o_a.free();
  }

  if(driverModus)
    fprintf(outputFile, "\n");
  else
    printf("\n");

  {
    const int Ntests = 20000;
    const int N[] = {1000 * 512, 2000 * 512, 4000 * 512, 8000 * 512};
    const int Nsize = sizeof(N) / sizeof(int);
    const int nWords = N[sizeof(N) / sizeof(N[0]) - 1];
    occa::memory o_a = device.malloc(nWords * sizeof(double));
    occa::memory o_b = device.malloc(nWords * sizeof(double));

    for(int i = 0; i < Nsize; ++i) {
      const long long int bytes = N[i] * sizeof(double);
      device.finish();
      timer::reset("memcpyDD");
      timer::tic("memcpyDD");
      for(int test = 0; test < Ntests; ++test) o_a.copyTo(o_b, bytes);
      device.finish();
      timer::toc("memcpyDD");
      timer::update();
      const double elapsed = timer::query("memcpyDD", "HOST:MAX") / Ntests;
      std::stringstream out;
      out << "D->D memcpy " <<  N[i] << " words, "
          << elapsed << " s, "
          << 2 * bytes / elapsed << " bytes/s\n";
      if(driverModus) {
        fprintf(outputFile, out.str().c_str());
        fflush(outputFile);
      } else {
        printf(out.str().c_str());
        fflush(stdout);
      }
    }
    o_a.free();
    o_b.free();
  }
  
  if(driverModus)
    fclose(outputFile);
    
}
