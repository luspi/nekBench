#include "omp.h"
#include "BP.hpp"

int solve(BP_t* BP,
          occa::memory &o_lambda,
          dfloat tol,
          occa::memory &o_r,
          occa::memory &o_x,
          double* opElapsed,
          bool driverModus,
          FILE *outputFile)
{
  mesh_t* mesh = BP->mesh;
  setupAide &options = BP->options;

  int Niter = 0;
  int maxIter = 1000;

  if(tol > 0) {
    options.setArgs("FIXED ITERATION COUNT", "FALSE");
  } else {
    options.setArgs("FIXED ITERATION COUNT", "TRUE");
    options.getArgs("MAXIMUM ITERATIONS", maxIter);
  }

  if(BP->allNeumann)
    BPZeroMean(BP, o_r);

  if(options.compareArgs("KRYLOV SOLVER", "PCG"))
    Niter = BPPCG(BP, o_lambda, o_r, o_x, tol, maxIter, opElapsed, driverModus, outputFile);

  if(BP->allNeumann)
    BPZeroMean(BP, o_x);

  return Niter;
}

void nekBone(setupAide &options, MPI_Comm mpiComm) {

  bool driverModus = options.compareArgs("DRIVER MODUS", "TRUE");

  // set up mesh stuff
  string fileName;
  int N, dim, elementType, kernelId;

  options.getArgs("POLYNOMIAL DEGREE", N);
  int cubN = 0;

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
  elementType = HEXAHEDRA;
  options.setArgs("ELLIPTIC INTEGRATION", "NODAL");
  options.setArgs("BASIS", "NODAL");
  options.getArgs("KERNEL ID", kernelId);

  int combineDot = 0;
  combineDot = 0; //options.compareArgs("COMBINE DOT PRODUCT", "TRUE");

  mesh_t* mesh;

  // set up mesh
  mesh = meshSetupBoxHex3D(N, cubN, options, mpiComm);
  mesh->elementType = elementType;

  // set up
  occa::properties kernelInfo;
  //kernelInfo["defines"].asObject();
  //kernelInfo["includes"].asArray();
  //kernelInfo["header"].asArray();
  //kernelInfo["flags"].asObject();

  meshOccaSetup3D(mesh, options, kernelInfo);

  FILE *outputFile;
  BP_t* BP = setup(mesh, kernelInfo, options, driverModus, &outputFile);

  dlong Ndofs = BP->Nfields * mesh->Np * mesh->Nelements;

  // default convergence tolerance
  dfloat tol = 1e-8;
  options.getArgs("SOLVER TOLERANCE", tol);

  int it;
  {
    double opElapsed = 0;
    int Ntests = 10;
    options.getArgs("NREPETITIONS", Ntests);

    // warm up  + correctness check
    BP->vecScaleKernel(BP->Nfields*BP->fieldOffset, 0.0, BP->o_x); // reset 
    BP->o_r.copyFrom(BP->r); // reset rhs
    it = solve(BP, BP->o_lambda, tol, BP->o_r, BP->o_x, &opElapsed);
    BP->o_x.copyTo(BP->x);
    const dlong offset = BP->fieldOffset;
    dfloat maxError = 0;
    for(dlong fld = 0; fld < BP->Nfields; ++fld)
      for(dlong e = 0; e < mesh->Nelements; ++e)
        for(int n = 0; n < mesh->Np; ++n) {
          dlong id = e * mesh->Np + n;
          dfloat xn = mesh->x[id];
          dfloat yn = mesh->y[id];
          dfloat zn = mesh->z[id];

          dfloat exact;
          double mode = 1.0;
          // hard coded to match the RHS used in BPSetup
          exact = cos(mode * M_PI * xn) * cos(
                  mode * M_PI * yn) * cos(mode * M_PI * zn);
          dfloat error = fabs(exact - BP->x[id + fld * offset]);
          maxError = mymax(maxError, error);
        }
    dfloat globalMaxError = 0;
    MPI_Allreduce(&maxError, &globalMaxError, 1, MPI_DFLOAT, MPI_MAX, mesh->comm);
    if(mesh->rank == 0) {
      std::stringstream out;
      out << "correctness check: maxError = " << globalMaxError << " in " << it << " iterations\n";
      if(driverModus)
        fprintf(outputFile, out.str().c_str());
      else
        std::cout << out.str();
    }

    if(options.compareArgs("FIXED ITERATION COUNT", "TRUE")) tol = 0;
    if(mesh->rank == 0) {
      if(driverModus)
        fprintf(outputFile, "\nrunning solver ...");
      else
        std::cout << std::endl << "running solver ..." << std::flush;
    }
    fflush(stdout);
    double elapsed = 0;
    for(int test = 0; test < Ntests; ++test) {
      BP->vecScaleKernel(BP->Nfields*BP->fieldOffset, 0.0, BP->o_x); // reset
      BP->o_r.copyFrom(BP->r); // reset rhs
      mesh->device.finish();
      MPI_Barrier(mesh->comm);
      double start = MPI_Wtime();
      it = solve(BP, BP->o_lambda, tol, BP->o_r, BP->o_x, &opElapsed);
      MPI_Barrier(mesh->comm);
      elapsed += MPI_Wtime() - start;
      timer::update();
    }
    if(mesh->rank == 0) {
      if(driverModus)
        fprintf(outputFile, " done\n");
      else
        std::cout << " done" << std::endl;
    }
    elapsed /= Ntests;

    // print statistics
    hlong globalNelements, localNelements = mesh->Nelements;
    MPI_Reduce(&localNelements, &globalNelements, 1, MPI_HLONG, MPI_SUM, 0, mesh->comm);

    hlong globalNdofs = pow(mesh->N,3) * mesh->Nelements; // mesh->Nlocalized;
    MPI_Allreduce(MPI_IN_PLACE, &globalNdofs, 1, MPI_HLONG, MPI_SUM, mesh->comm);
    const double gDOFs = BP->Nfields * (it * (globalNdofs / elapsed)) / 1.e9;

    const double Nlocal = mesh->Np * mesh->Nelements;
    const double gbytesPrecon = BP->Nfields*Nlocal;
    const double gbytesScaledAdd = 2. * BP->Nfields*Nlocal;
    double gbytesAx = (7 + 2 * BP->Nfields) * Nlocal;
    if(BP->BPid) gbytesAx += 2 * BP->Nfields*Nlocal;
    const double gbytesDot = (2 * BP->Nfields + 1) * Nlocal;
    const double gbytesPupdate = 4 * BP->Nfields*Nlocal;
    const double NGbytes = (gbytesPrecon + gbytesScaledAdd + gbytesAx + 3 * gbytesDot +  gbytesPupdate) * (sizeof(dfloat) / 1.e9);
    double bw = (it * NGbytes)/elapsed;
    MPI_Allreduce(MPI_IN_PLACE, &bw, 1, MPI_DFLOAT, MPI_SUM, mesh->comm);

    const double flopsPrecon = 0;
    const double flopsScaledAdd = 2 * BP->Nfields*Nlocal;
    double flopsAx = BP->Nfields*Nlocal*(12*mesh->Nq + 15);
    if(!BP->BPid) flopsAx += 5 * BP->Nfields*Nlocal;
    const double flopsDot = 3 * BP->Nfields*Nlocal;
    const double flopsPupdate = 4 * BP->Nfields*Nlocal;
    const double flops = flopsPrecon + flopsScaledAdd + flopsAx + 3*flopsDot + flopsPupdate;
    double gFlops = (it * flops)/elapsed/1e9;
    MPI_Allreduce(MPI_IN_PLACE, &gFlops, 1, MPI_DFLOAT, MPI_SUM, mesh->comm);

    double etime[10];
    if(BP->profiling) {
      etime[0] = timer::query("Ax", "DEVICE:MAX");
      etime[1] = timer::query("gs", "DEVICE:MAX");
      etime[2] = timer::query("updatePCG", "DEVICE:MAX");
      etime[3] = timer::query("dot", "DEVICE:MAX");
      etime[4] = timer::query("preco", "DEVICE:MAX");
      etime[5] = timer::query("Ax1", "DEVICE:MAX");
      etime[6] = timer::query("Ax2", "DEVICE:MAX");
      etime[7] = timer::query("AxGs", "DEVICE:MAX");
    }
    if(BP->overlap) {
      etime[0] = etime[5] + etime[6];
      etime[1] = etime[7] - etime[6];
    }
    
    if(mesh->rank == 0) {
      int knlId = 0;
      options.getArgs("KERNEL ID", knlId);
      
      std::stringstream out;

      int Nthreads =  omp_get_max_threads();
      out << "\nsummary\n"
          << "  MPItasks     : " << mesh->size << "\n";
      if(options.compareArgs("THREAD MODEL", "OPENMP"))
        out <<  "  OMPthreads   : " << Nthreads << "\n";
      out << "  polyN        : " << N << "\n"
          << "  Nelements    : " << globalNelements << "\n"
          << "  Nfields      : " << BP->Nfields << "\n"
          << "  iterations   : " << it << "\n"
          << "  Nrepetitions : " << Ntests << "\n"
          << "  elapsed time : " << Ntests * elapsed << " s\n"
          << "  throughput   : " << gDOFs << " GDOF/s/iter\n"
          << "  bandwidth    : " << bw << " GB/s\n"
          << "  GFLOPS/s     : " << gFlops << endl;

      if(BP->profiling)
        out << "\nbreakdown\n"
            << "  local Ax  : " << etime[0] << " s\n"
            << "  gs        : " << etime[1] << " s\n"
            << "  updatePCG : " << etime[2] << " s\n"
            << "  dot       : " << etime[3] << " s\n"
            << "  preco     : " << etime[4] << " s\n"
            << endl;
            
      if(driverModus) {
        fprintf(outputFile, out.str().c_str());
        fclose(outputFile);
      } else
        std::cout << out.str();
      
    }

  }

  BPDestroy(BP);
  meshDestroy(mesh);

}
