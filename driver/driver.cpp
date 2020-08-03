#include <string>
#include <iostream>
#include "../core/setCompilerFlags.hpp"
#include "../axhelm/axhelm.hpp"
#include "../bw/bw.hpp"
#include "../dot/dot.hpp"
#include "../gs/gs.hpp"
#include "../nekBone/nekBone.hpp"

void driver(std::string inifile) {
  
  std::cout << "reading ini file: " << inifile << std::endl;
  
  int mpiRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
  
  if(mpiRank == 0) {
    setupAide bwOptions;
    bwOptions.setArgs("THREAD MODEL", "SERIAL");
    bwOptions.setArgs("DRIVER MODUS", "TRUE");
    bw(bwOptions);
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  setupAide dotOptions;
  dotOptions.setArgs("N", "7");
  dotOptions.setArgs("N ELEMENTS", "1000");
  dotOptions.setArgs("THREAD MODEL", "SERIAL");
  dotOptions.setArgs("ARCH", "VOLTA");
  dotOptions.setArgs("N TESTS", "100");
  dotOptions.setArgs("DEVICE ID", "0");
  dotOptions.setArgs("BLOCK SIZE", "256");
  dotOptions.setArgs("GLOBAL", "0");
  dotOptions.setArgs("DRIVER MODUS", "TRUE");
  dot(dotOptions, MPI_COMM_WORLD);
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  setupAide gsOptions;
  gsOptions.setArgs("THREAD MODEL", "SERIAL");
  gsOptions.setArgs("N", "7");
  gsOptions.setArgs("NX", "10");
  gsOptions.setArgs("NY", "10");
  gsOptions.setArgs("NZ", "10");
  gsOptions.setArgs("OGS MODE", "0");
  gsOptions.setArgs("NTESTS", "100");
  gsOptions.setArgs("ENABLED TIMER", "0");
  gsOptions.setArgs("DUMMY KERNEL", "0");
  gsOptions.setArgs("FLOAT TYPE", "double");
  gsOptions.setArgs("GPUMPI", "0");
  gsOptions.setArgs("DEVICE NUMBER", "0");
  gsOptions.setArgs("DRIVER MODUS", "TRUE");
  gs(gsOptions, MPI_COMM_WORLD);
    
  MPI_Barrier(MPI_COMM_WORLD);
  
  if(mpiRank == 0) {
    setupAide axhelmOptions;
    axhelmOptions.setArgs("N", "7");
    axhelmOptions.setArgs("NDIM", "1");
    axhelmOptions.setArgs("NELEMENTS", "1000");
    axhelmOptions.setArgs("THREAD MODEL", "SERIAL");
    axhelmOptions.setArgs("ARCH", "VOLTA");
    axhelmOptions.setArgs("BKMODE", "1");
    axhelmOptions.setArgs("NTESTS", "100");
    axhelmOptions.setArgs("KERNELVERSION", "0");
    axhelmOptions.setArgs("DRIVER MODUS", "TRUE");
    axhelm(axhelmOptions);
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  setupAide nekBoneOptions(inifile);
  nekBoneOptions.setArgs("DRIVER MODUS", "TRUE");
  nekBone(nekBoneOptions);
  
}
