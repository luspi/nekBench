#include <string>
#include <iostream>
#include "../core/setCompilerFlags.hpp"
#include "../axhelm/axhelm.hpp"
#include "../bw/bw.hpp"
#include "../dot/dot.hpp"
#include "../allred/allred.hpp"
#include "../gs/gs.hpp"
#include "../nekBone/nekBone.hpp"
#include "parReader.hpp"

static std::vector<std::vector<libParanumal::setupAide> > options;

const std::vector<std::string> explode(const std::string& s, const char& c)
{
  std::string buff{""};
  std::vector<std::string> v;

  for(auto n:s) {
    if(n != c) buff+=n; else
    if(n == c && buff != "") { v.push_back(buff); buff = ""; }
  }
  if(buff != "") v.push_back(buff);

  return v;
}

void generateOptions(libParanumal::setupAide &inOpt, libParanumal::setupAide outOpt, std::vector<std::string> processed, int benchIndex) {

  std::vector<std::string> inKey = inOpt.getKeyword();
  for(size_t i = 0; i < inKey.size(); ++i) {
    // see if this one is already processed
    std::vector<std::string>::iterator it = std::find(processed.begin(), processed.end(), inKey[i]);
    // not yet:
    if(it == processed.end()) {

      std::vector<std::string> parts = explode(inOpt.getArgs(inKey[i]), *",");

      processed.push_back(inKey[i]);

      for(size_t j = 0; j < parts.size(); ++j) {

        size_t found = inKey[i].find("/");
        if(found != std::string::npos) {

          std::vector<std::string> optParts = explode(inKey[i], *"/");
          std::vector<std::string> valParts = explode(parts[j], *"/");

          for(int k = 0; k < optParts.size(); ++k)
            outOpt.setArgs(optParts[k], valParts[k]);

        } else
          outOpt.setArgs(inKey[i], parts[j]);

        generateOptions(inOpt, outOpt, processed, benchIndex);

      }

      return;

    }

  }

  options[benchIndex].push_back(outOpt);

}

void driver(std::string parfile, MPI_Comm comm) {

  int mpiRank, mpiSize;
  MPI_Comm_rank(comm, &mpiRank);
  MPI_Comm_size(comm, &mpiSize);

  int numBench = 7;

  std::vector<libParanumal::setupAide> masterOptions = parRead(parfile, comm);

  for(size_t i = 0; i < numBench; ++i)
    options.push_back(std::vector<libParanumal::setupAide>());

  for(size_t i = 0; i < numBench; ++i) {
    std::vector<std::string> processed;
    libParanumal::setupAide opt;
    generateOptions(masterOptions[i], opt, processed, i);
  }

  std::string benchmarks[numBench] = {"bw", "dot", "allreduce", "ogs", "pingpong", "axhelm", "nekbone"};
  for(int iBench = 0; iBench < numBench; ++iBench) {

    std::vector<libParanumal::setupAide> allopt = options[iBench];

    for(int iOpt = 0; iOpt < allopt.size(); ++iOpt) {

      bool enabled = (allopt[iOpt].compareArgs("ENABLED", "TRUE"));
      if(!enabled) break;

      int howManyMpiRanks;
      if(allopt[iOpt].compareArgs("MPI", "max") || allopt[iOpt].compareArgs("MPI", "MAX"))
        howManyMpiRanks = std::min(mpiSize, 512);
      else
        howManyMpiRanks = std::min(std::stoi(allopt[iOpt].getArgs("MPI")), 512);

      MPI_Comm subComm;

      if(howManyMpiRanks == mpiSize) {
        MPI_Comm_dup(comm, &subComm);
      } else {

        MPI_Group worldGroup;
        MPI_Comm_group(comm, &worldGroup);

        int listRanks[howManyMpiRanks];
        for(int i = 0; i < howManyMpiRanks; ++i)
          listRanks[i] = i;

        MPI_Group subGroup;
        MPI_Group_incl(worldGroup, howManyMpiRanks, listRanks, &subGroup);

        MPI_Comm_create_group(comm, subGroup, 0, &subComm);

      }

      if(mpiRank < howManyMpiRanks) {
        if(benchmarks[iBench] == "bw") {
          bw(allopt[iOpt], subComm);
        } else if(benchmarks[iBench] == "dot") {
          dot(allopt[iOpt], subComm);
        } else if(benchmarks[iBench] == "allreduce") {
          allred(allopt[iOpt], subComm);
        } else if(benchmarks[iBench] == "ogs") {
          gs(allopt[iOpt], subComm, true, false);
        } else if(benchmarks[iBench] == "pingpong") {
          gs(allopt[iOpt], subComm, false, true);
        } else if(benchmarks[iBench] == "axhelm") {
          axhelm(allopt[iOpt], subComm);
        } else if(benchmarks[iBench] == "nekbone") {
          nekBone(allopt[iOpt], subComm);
        }
      }

      MPI_Barrier(comm);

    }

  }
  
}
