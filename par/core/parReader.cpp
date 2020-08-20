#include <cstdio>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <getopt.h>

#include "tinyexpr.h"
#include "inipp.hpp"

#include "nrs.hpp"
#include "bcMap.hpp"

#define exit(a,b)  { if(rank == 0) cout << a << endl; EXIT(1); }
#define UPPER(a)  { transform(a.begin(), a.end(), a.begin(), std::ptr_fun<int, int>(std::toupper)); \
}
#define LOWER(a)  { transform(a.begin(), a.end(), a.begin(), std::ptr_fun<int, int>(std::tolower)); \
}

void setDefaultSettings(std::vector<libParanumal::setupAide> &options)
{

  options[0].setArgs("FORMAT", string("1.0"));
  options[0].setArgs("VERBOSE", string("FALSE"));
  options[0].setArgs("DRIVER MODUS", string("TRUE"));
  options[0].setArgs("ENABLED", string("TRUE"));
  options[0].setArgs("MPI", string("1"));
  options[0].setArgs("THREAD MODEL", string("CUDA,SERIAL"));

  options[1].setArgs("FORMAT", string("1.0"));
  options[1].setArgs("VERBOSE", string("FALSE"));
  options[1].setArgs("DRIVER MODUS", string("TRUE"));
  options[1].setArgs("ENABLED", string("TRUE"));
  options[1].setArgs("MPI", string("1,MAX"));
  options[1].setArgs("N", string("3,7,9"));
  options[1].setArgs("N ELEMENTS", string("2048,4096,8192,16384"));
  options[1].setArgs("THREAD MODEL", string("SERIAL,CUDA"));
  options[1].setArgs("ARCH", string("VOLTA"));
  options[1].setArgs("N TESTS", string("100"));
  options[1].setArgs("DEVICE ID", string("0"));
  options[1].setArgs("BLOCK SIZE", string("256"));
  options[1].setArgs("GLOBAL", string("0"));

  options[2].setArgs("FORMAT", string("1.0"));
  options[2].setArgs("VERBOSE", string("FALSE"));
  options[2].setArgs("DRIVER MODUS", string("TRUE"));
  options[2].setArgs("ENABLED", string("TRUE"));
  options[2].setArgs("MPI", string("MAX"));
  options[2].setArgs("THREAD MODEL", string("SERIAL,CUDA"));

  options[3].setArgs("FORMAT", string("1.0"));
  options[3].setArgs("VERBOSE", string("FALSE"));
  options[3].setArgs("DRIVER MODUS", string("TRUE"));
  options[3].setArgs("ENABLED", string("TRUE"));
  options[3].setArgs("MPI", string("MAX"));
  options[3].setArgs("THREAD MODEL", string("CUDA"));
  options[3].setArgs("N", string("3,7,9"));
  options[3].setArgs("NX/NY/NZ", string("8/16/16,16/16/16,16/16/32,16/32/32"));
  options[3].setArgs("OGS MODE", string("-1"));
  options[3].setArgs("NTESTS", string("100"));
  options[3].setArgs("ENABLED TIMER", string("0"));
  options[3].setArgs("DUMMY KERNEL", string("0"));
  options[3].setArgs("FLOAT TYPE", string("double"));
  options[3].setArgs("GPUMPI", string("0"));
  options[3].setArgs("DEVICE NUMBER", string("0"));

  options[4].setArgs("FORMAT", string("1.0"));
  options[4].setArgs("VERBOSE", string("FALS"));
  options[4].setArgs("DRIVER MODUS", string("TRUE"));
  options[4].setArgs("ENABLED", string("TRUE"));
  options[4].setArgs("MPI", string("MAX"));
  options[4].setArgs("THREAD MODEL", string("SERIAL,CUDA"));
  options[4].setArgs("N", string("3"));
  options[4].setArgs("NX", string("8"));
  options[4].setArgs("NY", string("16"));
  options[4].setArgs("NZ", string("16"));
  options[4].setArgs("DEVICE NUMBER", string("0"));
  options[4].setArgs("ENABLED TIMER", string("0"));
  options[4].setArgs("FLOAT TYPE", string("double"));
  options[4].setArgs("GPUMPI", string("0"));

  options[5].setArgs("FORMAT", string("1.0"));
  options[5].setArgs("VERBOSE", string("FALSE"));
  options[5].setArgs("DRIVER MODUS", string("TRUE"));
  options[5].setArgs("ENABLED", string("TRUE"));
  options[5].setArgs("MPI", string("1"));
  options[5].setArgs("N", string("3,7,9"));
  options[5].setArgs("NDIM", string("1,3"));
  options[5].setArgs("NELEMENTS", string("2048,4096,8192,16384"));
  options[5].setArgs("THREAD MODEL", string("SERIAL,CUDA"));
  options[5].setArgs("ARCH", string("VOLTA"));
  options[5].setArgs("BKMODE", string("1"));
  options[5].setArgs("NTESTS", string("1000"));
  options[5].setArgs("KERNELVERSION", string("0"));

  options[6].setArgs("FORMAT", string("1.0"));
  options[6].setArgs("VERBOSE", string("FALSE"));
  options[6].setArgs("DRIVER MODUS", string("TRUE"));
  options[6].setArgs("ENABLED", string("TRUE"));
  options[6].setArgs("MPI", string("1,MAX"));
  options[6].setArgs("BPMODE", string("TRUE"));
  options[6].setArgs("NFIELDS", string("1,3"));
  options[6].setArgs("OVERLAP", string("TRUE"));
  options[6].setArgs("LAMBDA", string("1.0"));
  options[6].setArgs("POLYNOMIAL DEGREE", string("3,7,9"));
  options[6].setArgs("NX/NY/NZ", string("8/16/16,16/16/16,16/16/32,16/32/32"));
  options[6].setArgs("KERNEL ID", string("0"));
  options[6].setArgs("PLATFORM NUMBER", string("0"));
  options[6].setArgs("DEVICE NUMBER", string("0"));
  options[6].setArgs("ARCH/THREAD MODEL", string("VOLTA/CUDA,CPU/NATIVE+SERIAL"));
  options[6].setArgs("KRYLOV SOLVER", string("PCG"));
  options[6].setArgs("PRECONDITIONER", string("COPY"));
  options[6].setArgs("SOLVER TOLERANCE", string("1e-6"));
  options[6].setArgs("NREPETITIONS", string("1"));
  options[6].setArgs("MAXIMUM ITERATIONS", string("5000"));
  options[6].setArgs("FIXED ITERATION COUNT", string("TRUE"));
  options[6].setArgs("PROFILING", string("FALSE,TRUE"));
  options[6].setArgs("TIMER SYNC", string("TRUE"));

}

std::vector<libParanumal::setupAide> parRead(std::string &setupFile, MPI_Comm comm)
{
  int rank;
  MPI_Comm_rank(comm, &rank);

  // set default options
  std::vector<libParanumal::setupAide> options;
  options.resize(7);
  setDefaultSettings(options);

  // if no par file was specified, stop here
  if(setupFile == "")
    return options;

  ////////////////

  const char* ptr = realpath(setupFile.c_str(), NULL);
  if (!ptr) {
    if (rank == 0) cout << "\nERROR: Cannot find " << setupFile << "!\n";
    ABORT(1);
  }

  char* rbuf;
  long fsize;
  if(rank == 0) {
    FILE* f = fopen(setupFile.c_str(), "rb");
    fseek(f, 0, SEEK_END);
    fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    rbuf = new char[fsize];
    fread(rbuf, 1, fsize, f);
    fclose(f);
  }
  MPI_Bcast(&fsize, sizeof(fsize), MPI_BYTE, 0, comm);
  if(rank != 0) rbuf = new char[fsize];
  MPI_Bcast(rbuf, fsize, MPI_CHAR, 0, comm);
  stringstream is;
  is.write(rbuf, fsize);

  inipp::Ini<char> ini;
  ini.parse(is);
  ini.interpolate();


  // check general options

  bool verbose = false;
  if(ini.extract("general", "verbose", verbose))
    for(int i = 0; i < 7; ++i) options[i].setArgs("VERBOSE", (verbose ? "TRUE" : "FALSE"));

  bool drivermodus = false;
  if(ini.extract("general", "driver modus", drivermodus))
    for(int i = 0; i < 7; ++i) options[i].setArgs("DRIVER MODUS", (drivermodus ? "TRUE" : "FALSE"));


  // check bw options

  bool bwEnabled = true;
  if(ini.extract("bw", "enabled", bwEnabled))
    options[0].setArgs("ENABLED", (bwEnabled ? "TRUE" : "FALSE"));

  std::string bwMPI = "";
  if(ini.extract("bw", "mpi", bwMPI))
    options[0].setArgs("MPI", bwMPI);

  std::string bwTM = "";
  if(ini.extract("bw", "thread model", bwTM)) {
    UPPER(bwTM);
    options[0].setArgs("THREAD MODEL", bwTM);
  }


  // check dot options

  bool dotEnabled = true;
  if(ini.extract("dot", "enabled", dotEnabled))
    options[1].setArgs("ENABLED", (dotEnabled ? "TRUE" : "FALSE"));

  std::string dotMPI = "";
  if(ini.extract("dot", "mpi", dotMPI))
    options[1].setArgs("MPI", dotMPI);

  std::string dotN = "";
  if(ini.extract("dot", "n", dotN))
    options[1].setArgs("N", dotN);

  std::string dotNele = "";
  if(ini.extract("dot", "n elements", dotNele))
    options[1].setArgs("N ELEMENTS", dotNele);

  std::string dotTM = "";
  if(ini.extract("dot", "thread model", dotTM)) {
    UPPER(dotTM);
    options[1].setArgs("THREAD MODEL", dotTM);
  }

  std::string dotArch = "";
  if(ini.extract("dot", "arch", dotArch)) {
    UPPER(dotArch);
    options[1].setArgs("ARCH", dotArch);
  }

  std::string dotNTests = "";
  if(ini.extract("dot", "n tests", dotNTests))
    options[1].setArgs("N TESTS", dotNTests);

  std::string dotDID = "";
  if(ini.extract("dot", "device id", dotDID))
    options[1].setArgs("DEVICE ID", dotDID);

  std::string dotBS = "";
  if(ini.extract("dot", "block size", dotBS))
    options[1].setArgs("BLOCK SIZE", dotBS);

  std::string dotGlob = "";
  if(ini.extract("dot", "global", dotGlob))
    options[1].setArgs("GLOBAL", dotGlob);


  // check allreduce options

  bool allredEnabled = true;
  if(ini.extract("allreduce", "enabled", allredEnabled))
    options[2].setArgs("ENABLED", (allredEnabled ? "TRUE" : "FALSE"));

  std::string allredMPI = "";
  if(ini.extract("allreduce", "mpi", allredMPI))
    options[2].setArgs("MPI", allredMPI);

  std::string allredTM = "";
  if(ini.extract("allreduce", "thread model", allredTM)) {
    UPPER(allredTM);
    options[2].setArgs("THREAD MODEL", allredTM);
  }


  // check ogs options

  bool ogsEnabled = true;
  if(ini.extract("ogs", "enabled", ogsEnabled))
    options[3].setArgs("ENABLED", (ogsEnabled ? "TRUE" : "FALSE"));

  std::string ogsMPI = "";
  if(ini.extract("ogs", "mpi", ogsMPI))
    options[3].setArgs("MPI", ogsMPI);

  std::string ogsTM = "";
  if(ini.extract("ogs", "thread model", ogsTM)) {
    UPPER(ogsTM);
    options[3].setArgs("THREAD MODEL", ogsTM);
  }

  std::string ogsN = "";
  if(ini.extract("ogs", "n", ogsN))
    options[3].setArgs("N", ogsN);

  std::string ogsNXYZ = "";
  if(ini.extract("ogs", "nx/ny/nz", ogsNXYZ))
    options[3].setArgs("NX/NY/NZ", ogsNXYZ);

  std::string ogsOM = "";
  if(ini.extract("ogs", "ogs mode", ogsOM))
    options[3].setArgs("OGS MODE", ogsOM);

  std::string ogsNT = "";
  if(ini.extract("ogs", "n tests", ogsNT))
    options[3].setArgs("NTESTS", ogsNT);

  std::string ogsET = "";
  if(ini.extract("ogs", "enabled timer", ogsET))
    options[3].setArgs("ENABLED TIMER", ogsET);

  std::string ogsDK = "";
  if(ini.extract("ogs", "dummy kernel", ogsDK))
    options[3].setArgs("DUMMY KERNEL", ogsDK);

  std::string ogsFT = "";
  if(ini.extract("ogs", "float type", ogsFT))
    options[3].setArgs("FLOAT TYPE", ogsFT);

  std::string ogsGPUMPI = "";
  if(ini.extract("ogs", "gpumpi", ogsGPUMPI))
    options[3].setArgs("GPUMPI", ogsGPUMPI);

  std::string ogsDN = "";
  if(ini.extract("ogs", "device number", ogsDN))
    options[3].setArgs("DEVICE NUMBER", ogsDN);


  // check pingpong options

  bool ppEnabled = true;
  if(ini.extract("pingpong", "enabled", ppEnabled))
    options[4].setArgs("ENABLED", (ppEnabled ? "TRUE" : "FALSE"));

  std::string ppMPI = "";
  if(ini.extract("pingpong", "mpi", ppMPI))
    options[4].setArgs("MPI", ppMPI);

  std::string ppTM = "";
  if(ini.extract("pingpong", "thread model", ppTM)) {
    UPPER(ppTM);
    options[4].setArgs("THREAD MODEL", ppTM);
  }

  std::string ppN = "";
  if(ini.extract("pingpong", "n", ppN))
    options[4].setArgs("N", ppN);

  std::string ppNXYZ = "";
  if(ini.extract("pingpong", "nx/ny/nz", ppNXYZ))
    options[4].setArgs("NX/NY/NZ", ppNXYZ);

  std::string ppDN = "";
  if(ini.extract("pingpong", "device number", ppDN))
    options[4].setArgs("DEVICE NUMBER", ppDN);

  std::string ppET = "";
  if(ini.extract("pingpong", "enabled timer", ppET))
    options[4].setArgs("ENABLED TIMER", ppET);

  std::string ppFT = "";
  if(ini.extract("pingpong", "float type", ppFT))
    options[4].setArgs("FLOAT TYPE", ppFT);

  std::string ppGPUMPI = "";
  if(ini.extract("pingpong", "gpumpi", ppGPUMPI))
    options[4].setArgs("GPUMPI", ppGPUMPI);


  // check axhelm options

  bool axEnabled = true;
  if(ini.extract("axhelm", "enabled", axEnabled))
    options[5].setArgs("ENABLED", (axEnabled ? "TRUE" : "FALSE"));

  std::string axMPI = "";
  if(ini.extract("axhelm", "mpi", axMPI))
    options[5].setArgs("MPI", axMPI);

  std::string axN = "";
  if(ini.extract("axhelm", "n", axN))
    options[5].setArgs("N", axN);

  std::string axNDIM = "";
  if(ini.extract("axhelm", "ndim", axNDIM))
    options[5].setArgs("NDIM", axNDIM);

  std::string axEle = "";
  if(ini.extract("axhelm", "n elements", axEle))
    options[5].setArgs("NELEMENTS", axEle);

  std::string axTM = "";
  if(ini.extract("axhelm", "thread model", axTM)) {
    UPPER(axTM);
    options[5].setArgs("THREAD MODEL", axTM);
  }

  std::string axARCH = "";
  if(ini.extract("axhelm", "arch", axARCH)) {
    UPPER(axARCH);
    options[5].setArgs("ARCH", axARCH);
  }

  std::string axBKMODE = "";
  if(ini.extract("axhelm", "bkmode", axBKMODE))
    options[5].setArgs("BKMODE", axBKMODE);

  std::string axNTESTS = "";
  if(ini.extract("axhelm", "n tests", axNTESTS))
    options[5].setArgs("NTESTS", axNTESTS);

  std::string axKV = "";
  if(ini.extract("axhelm", "kernel version", axKV))
    options[5].setArgs("KERNELVERSION", axKV);


  // check nekbone settings

  bool nbEnabled = true;
  if(ini.extract("nekbone", "enabled", nbEnabled))
    options[6].setArgs("ENABLED", (nbEnabled ? "TRUE" : "FALSE"));

  std::string nbMPI = "";
  if(ini.extract("nekbone", "mpi", nbMPI))
    options[6].setArgs("MPI", nbMPI);

  std::string nbBP = "";
  if(ini.extract("nekbone", "bpmode", nbBP))
    options[6].setArgs("BPMODE", nbBP);

  std::string nbNF = "";
  if(ini.extract("nekbone", "n fields", nbNF))
    options[6].setArgs("NFIELDS", nbNF);

  std::string nbOV = "";
  if(ini.extract("nekbone", "overlap", nbOV))
    options[6].setArgs("OVERLAP", nbOV);

  std::string nbL = "";
  if(ini.extract("nekbone", "lambda", nbL))
    options[6].setArgs("LAMBDA", nbL);

  std::string nbPD = "";
  if(ini.extract("nekbone", "polynomial degree", nbPD))
    options[6].setArgs("POLYNOMIAL DEGREE", nbPD);

  std::string nbNXYZ = "";
  if(ini.extract("nekbone", "nx/ny/nz", nbNXYZ))
    options[6].setArgs("NX/NY/NZ", nbNXYZ);

  std::string nbKID = "";
  if(ini.extract("nekbone", "kernel id", nbKID))
    options[6].setArgs("KERNEL ID", nbKID);

  std::string nbPN = "";
  if(ini.extract("nekbone", "platform number", nbPN))
    options[6].setArgs("PLATFORM NUMBER", nbPN);

  std::string nbDN = "";
  if(ini.extract("nekbone", "device number", nbDN))
    options[6].setArgs("DEVICE NUMBER", nbDN);

  std::string nbATM = "";
  if(ini.extract("nekbone", "arch/thread model", nbATM)) {
    UPPER(nbATM);
    options[6].setArgs("ARCH/THREAD MODEL", nbATM);
  }

  std::string nbKrylov = "";
  if(ini.extract("nekbone", "krylov solver", nbKrylov)) {
    UPPER(nbKrylov);
    options[6].setArgs("KRYLOV SOLVER", nbKrylov);
  }

  std::string nbPrec = "";
  if(ini.extract("nekbone", "preconditioner", nbPrec)) {
    UPPER(nbPrec);
    options[6].setArgs("PRECONDITIONER", nbPrec);
  }

  std::string nbTol = "";
  if(ini.extract("nekbone", "solver tolerance", nbTol))
    options[6].setArgs("SOLVER TOLERANCE", nbTol);

  std::string nbRep = "";
  if(ini.extract("nekbone", "n repetitions", nbRep))
    options[6].setArgs("NREPETITIONS", nbRep);

  std::string nbMaxIter = "";
  if(ini.extract("nekbone", "maximum iterations", nbMaxIter))
    options[6].setArgs("MAXIMUM ITERATIONS", nbMaxIter);

  std::string nbFixIt = "";
  if(ini.extract("nekbone", "fixed iteration count", nbFixIt))
    options[6].setArgs("FIXED ITERATION COUNT", nbFixIt);

  std::string nbProf = "";
  if(ini.extract("nekbone", "profiling", nbProf))
    options[6].setArgs("PROFILING", nbProf);

  std::string nbTS = "";
  if(ini.extract("nekbone", "timer sync", nbTS))
    options[6].setArgs("TIMER SYNC", nbTS);


  return options;

}
