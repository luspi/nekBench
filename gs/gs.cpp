#include <math.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>

#include <list>
#include <algorithm>

#include "mpi.h"
#include "params.h"
#include "mesh.h"
#include "timer.hpp"

#include "ping_pong.h"

#include "mygs.h"

void gs(setupAide &options, MPI_Comm mpiComm, bool testOgsModes, bool testPingPong) {

  int rank;
  MPI_Comm_rank (mpiComm, &rank);

  bool driverModus = options.compareArgs("DRIVER MODUS", "TRUE");

  const int N = std::stoi(options.getArgs("N"));
  const int NX = std::stoi(options.getArgs("NX"));
  const int NY = std::stoi(options.getArgs("NY"));
  const int NZ = std::stoi(options.getArgs("NZ"));

  std::list<ogs_mode> ogs_mode_list;
  if(testOgsModes) {
    int mode = std::stoi(options.getArgs("OGS MODE"));
    if(mode > 0) {
      if(mode > 4) {
        if(rank == 0) printf("invalid ogs_mode!\n");
        MPI_Abort(mpiComm,1);
      }
      if(mode) ogs_mode_list.push_back((ogs_mode)(mode - 1));
    } else {
      ogs_mode_list.push_back(OGS_DEFAULT);
      ogs_mode_list.push_back(OGS_HOSTMPI);
    }
  }

  int enabledTimer = std::stoi(options.getArgs("ENABLED TIMER"));
  std::string floatType = options.getArgs("FLOAT TYPE");
  int unit_size = (floatType == "float" ? sizeof(float) : sizeof(double));
  int enabledGPUMPI = std::stoi(options.getArgs("GPUMPI"));
  if(enabledGPUMPI) ogs_mode_list.push_back(OGS_DEVICEMPI);

  options.setArgs("BOX NX", std::to_string(NX));
  options.setArgs("BOX NY", std::to_string(NY));
  options.setArgs("BOX NZ", std::to_string(NZ));

  options.setArgs("BOX XMIN", "-1.0");
  options.setArgs("BOX YMIN", "-1.0");
  options.setArgs("BOX ZMIN", "-1.0");
  options.setArgs("BOX XMAX", "1.0");
  options.setArgs("BOX YMAX", "1.0");
  options.setArgs("BOX ZMAX", "1.0");
  options.setArgs("MESH DIMENSION", "3");
  options.setArgs("BOX DOMAIN", "TRUE");

  options.setArgs("DISCRETIZATION", "CONTINUOUS");
  options.setArgs("ELEMENT MAP", "ISOPARAMETRIC");
  options.setArgs("ELEMENT TYPE", std::to_string(HEXAHEDRA));
  options.setArgs("POLYNOMIAL DEGREE", std::to_string(N));

  // setup mesh
  mesh_t* mesh = meshSetupBoxHex3D(N, 0, options);
  mesh->elementType = HEXAHEDRA;
  occa::properties kernelInfo;
  meshOccaSetup3D(mesh, options, kernelInfo);

  // setup dummy kernel
  occa::kernel kernel;
  if(testOgsModes) {
    int dummyKernel = std::stoi(options.getArgs("DUMMY KERNEL"));
    if(dummyKernel) {
      for (int r = 0; r < 2; r++) {
        if ((r == 0 && mesh->rank == 0) || (r == 1 && mesh->rank > 0))
          kernel = mesh->device.buildKernel("gs_dummy.okl", "dummy", kernelInfo);
        MPI_Barrier(mpiComm);
      }
    }
  }

  timer::init(mesh->comm, mesh->device, 0);

  // setup gs
  const dlong Nlocal = mesh->Nelements * mesh->Np;
  ogs_t* ogs = ogsSetup(Nlocal, mesh->globalIds, mesh->comm, 1, mesh->device);
  mygsSetup(ogs, enabledTimer);
  mesh->ogs = ogs;

  if(!driverModus)
    meshPrintPartitionStatistics(mesh);

  double* U = (double*) calloc(Nlocal, sizeof(double));
  occa::memory o_U = mesh->device.malloc(Nlocal * sizeof(double), U);

  occa::memory o_q = mesh->device.malloc(Nlocal * unit_size);

  // warm-up + check for correctness
  for (auto const& ogs_mode_enum : ogs_mode_list) {
    if(floatType.compare("float") == 0) {
      float* Q = (float*) calloc(Nlocal, unit_size);
      for(int i = 0; i < Nlocal; i++) Q[i] = 1;
      o_q.copyFrom(Q, Nlocal * unit_size);
      free(Q);
    } else {
      double* Q = (double*) calloc(Nlocal, unit_size);
      for(int i = 0; i < Nlocal; i++) Q[i] = 1;
      o_q.copyFrom(Q, Nlocal * unit_size);
      free(Q);
    }

    mygsStart(o_q, floatType.c_str(), ogsAdd, ogs, ogs_mode_enum);
    mygsFinish(o_q, floatType.c_str(), ogsAdd, ogs, ogs_mode_enum);

    if(floatType.compare("float") == 0) {
      float* Q = (float*) calloc(Nlocal, unit_size);
      o_q.copyTo(Q, Nlocal * unit_size);
      for(int i = 0; i < Nlocal; i++) Q[i] = 1 / Q[i];
      o_q.copyFrom(Q, Nlocal * unit_size);
      free(Q);
    } else {
      double* Q = (double*) calloc(Nlocal, unit_size);
      o_q.copyTo(Q, Nlocal * unit_size);
      for(int i = 0; i < Nlocal; i++) Q[i] = 1 / Q[i];
      o_q.copyFrom(Q, Nlocal * unit_size);
      free(Q);
    }

    mygsStart(o_q, floatType.c_str(), ogsAdd, ogs, ogs_mode_enum);
    mygsFinish(o_q, floatType.c_str(), ogsAdd, ogs, ogs_mode_enum);

    long long int nPts = 0;
    if(floatType.compare("float") == 0) {
      float* Q = (float*) calloc(Nlocal, unit_size);
      o_q.copyTo(Q, Nlocal * unit_size);
      for(int i = 0; i < Nlocal; i++) nPts += (long long int)std::nearbyint(Q[i]);
      free(Q);
    } else {
      double* Q = (double*) calloc(Nlocal, unit_size);
      o_q.copyTo(Q, Nlocal * unit_size);
      for(int i = 0; i < Nlocal; i++) {
        double tmp = std::nearbyint(Q[i]);
        nPts += (long long int)tmp;
        //if(tmp != 1) printf("here %g\n",tmp);
      }
      free(Q);
    }
    MPI_Allreduce(MPI_IN_PLACE,&nPts,1,MPI_LONG_LONG_INT,MPI_SUM,mesh->comm);
    if(nPts - NX * NY * NZ * (long long int)mesh->Np != 0) {
      if(mesh->rank == 0) printf("\ncorrectness check failed for mode=%d! %ld\n",
                                 ogs_mode_enum,
                                 nPts);
      fflush(stdout);
      //MPI_Abort(mesh->comm, 1);
    }
  }

  if(mesh->rank == 0 && !driverModus) cout << "\nstarting measurement ...\n";
  fflush(stdout);

  if(testPingPong) {

    // ping pong
    mesh->device.finish();
    MPI_Barrier(mpiComm);
    {
      const int nPairs = mesh->size / 2;
      pingPongMulti(nPairs, 0, mesh->device, mpiComm, driverModus);
      pingPongSingle(0, mesh->device, mpiComm, driverModus);
      if(enabledGPUMPI) {
        pingPongMulti(nPairs, enabledGPUMPI, mesh->device, mpiComm, driverModus);
        pingPongSingle(1, mesh->device, mpiComm, driverModus);
      }
    }

  }

  if(testOgsModes) {

    int Ntests = std::stoi(options.getArgs("NTESTS"));
    int dummyKernel = std::stoi(options.getArgs("DUMMY KERNEL"));

    for (auto const& ogs_mode_enum : ogs_mode_list) {
      occa::stream defaultStream = mesh->device.getStream();
      occa::stream kernelStream  = mesh->device.createStream();

      // gs
      timer::reset();
      mesh->device.setStream(kernelStream);
      mesh->device.finish();
      MPI_Barrier(mesh->comm);
      const double start = MPI_Wtime();
      for(int test = 0; test < Ntests; ++test) {
        mygsStart(o_q, floatType.c_str(), ogsAdd, ogs, ogs_mode_enum);
        if(dummyKernel) {
          if(enabledTimer) timer::tic("dummyKernel");
          kernel(Nlocal, o_U);
          mesh->device.setStream(defaultStream);
          if(enabledTimer) timer::toc("dummyKernel");
        }
        mygsFinish(o_q, floatType.c_str(), ogsAdd, ogs, ogs_mode_enum);
        if(enabledTimer) timer::update();
      }
      mesh->device.finish();
      mesh->device.setStream(defaultStream);
      MPI_Barrier(mesh->comm);
      const double elapsed = (MPI_Wtime() - start) / Ntests;

      // print stats
      const int Ntimer = 10;
      double etime[Ntimer];
      if(enabledTimer) {
        etime[0] = timer::query("gather_halo", "DEVICE:MAX");
        etime[1] = timer::query("gs_interior", "DEVICE:MAX");
        etime[2] = timer::query("scatter", "DEVICE:MAX");
        etime[3] = timer::query("gs_host", "HOST:MAX");
        etime[4] = timer::query("gs_memcpy_dh", "DEVICE:MAX");
        etime[5] = timer::query("gs_memcpy_hd", "DEVICE:MAX");
        etime[6] = timer::query("dummyKernel", "DEVICE:MAX");
        etime[7] = timer::query("pw_exec", "HOST:MAX");
        etime[8] = timer::query("pack", "DEVICE:MAX");
        etime[9] = timer::query("unpack", "DEVICE:MAX");
        for(int i = 0; i < Ntimer; i++) etime[i] = std::max(etime[i],0.0) / Ntests;
      }

      if(mesh->rank == 0 && !driverModus) {
        int Nthreads =  omp_get_max_threads();
        cout << "\nsummary\n"
            << "  ogsMode                       : " << ogs_mode_enum << "\n"
            << "  MPItasks                      : " << mesh->size << "\n";

        if(options.compareArgs("THREAD MODEL", "OPENMP"))
          cout << "  OMPthreads                    : " << Nthreads << "\n";

        cout << "  polyN                         : " << N << "\n"
            << "  Nelements                     : " << NX * NY * NZ << "\n"
            << "  Nrepetitions                  : " << Ntests << "\n"
            << "  floatType                     : " << floatType << "\n"
            << "  throughput                    : " <<
        ((double)(NX * NY * NZ) * N * N * N / elapsed) / 1.e9 << " GDOF/s\n"
            << "  avg elapsed time              : " << elapsed << " s\n";

        if(enabledTimer) {
          cout << "    gather halo                 : " << etime[0] << " s\n"
              << "    gs interior                 : " << etime[1] << " s\n"
              << "    scatter halo                : " << etime[2] << " s\n"
              << "    memcpy host<->device        : " << etime[4] + etime[5] << " s\n";

          if(ogs_mode_enum == OGS_DEFAULT)
            cout << "    gslib_host                  : " << etime[3] << " s\n";
          else
            cout << "    pack/unpack buf             : " << etime[8] + etime[9] << " s\n"
                << "    pw exec                     : " << etime[7] << " s\n";

          cout << "  avg elapsed time dummy kernel : " << etime[6] << " s\n";
        }
      }
    }
  }
    
  free(U);
  o_U.free();
  o_q.free();
  if(ogs->haloGatherOffsets) free(ogs->haloGatherOffsets);
  if(ogs->haloGatherIds) free(ogs->haloGatherIds);
  if(ogs->haloGshSym) free(ogs->haloGshSym);
  if(ogs->haloGshNonSym) free(ogs->haloGshNonSym);
  if(ogs->localGatherOffsets) free(ogs->localGatherOffsets);
  if(ogs->gatherInvDegree) free(ogs->gatherInvDegree);
  if(ogs->localGatherIds) free(ogs->localGatherIds);
  if(ogs->invDegree) free(ogs->invDegree);
  // Calling ogsFree() on ogs causes a crash.
  // As much as possible of the memory inside ogs is freed manually, so that we simply resort to decreasing the ref counter here.
  ogs::Nrefs--;
  meshDestroy(mesh);

}
