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
static std::vector<std::vector<std::string> > optionsThatVary;

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

          for(int k = 0; k < optParts.size(); ++k) {
            outOpt.setArgs(optParts[k], valParts[k]);
            if(j == 0 && parts.size() > 1) {
              // only store this option if it is not already stored
              // this can happen if this key is part of the default filename options set below
              std::vector<std::string>::iterator it = std::find(optionsThatVary[benchIndex].begin(), optionsThatVary[benchIndex].end(), optParts[k]);
              if(it == optionsThatVary[benchIndex].end())
                optionsThatVary[benchIndex].push_back(optParts[k]);
            }
          }

        } else {
          outOpt.setArgs(inKey[i], parts[j]);
          if(j == 0 && parts.size() > 1) {
            std::vector<std::string>::iterator it = std::find(optionsThatVary[benchIndex].begin(), optionsThatVary[benchIndex].end(), inKey[i]);
            if(it == optionsThatVary[benchIndex].end())
              optionsThatVary[benchIndex].push_back(inKey[i]);
          }
        }

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
  optionsThatVary.resize(masterOptions.size());

  for(size_t i = 0; i < numBench; ++i)
    options.push_back(std::vector<libParanumal::setupAide>());

  // these options will always be put into the filename even if they don't vary
  optionsThatVary[0].push_back("THREAD MODEL");
  optionsThatVary[1].push_back("THREAD MODEL");
  optionsThatVary[1].push_back("ARCH");
  optionsThatVary[1].push_back("N");
  optionsThatVary[1].push_back("N ELEMENTS");
  optionsThatVary[2].push_back("THREAD MODEL");
  optionsThatVary[3].push_back("THREAD MODEL");
  optionsThatVary[3].push_back("N");
  optionsThatVary[3].push_back("NX");
  optionsThatVary[3].push_back("NY");
  optionsThatVary[3].push_back("NZ");
  optionsThatVary[4].push_back("THREAD MODEL");
  optionsThatVary[4].push_back("MPI");
  optionsThatVary[5].push_back("THREAD MODEL");
  optionsThatVary[5].push_back("N");
  optionsThatVary[5].push_back("NDIM");
  optionsThatVary[5].push_back("NELEMENTS");
  optionsThatVary[6].push_back("THREAD MODEL");
  optionsThatVary[6].push_back("ARCH");
  optionsThatVary[6].push_back("NFIELDS");
  optionsThatVary[6].push_back("POLYNOMIAL DEGREE");
  optionsThatVary[6].push_back("NX");
  optionsThatVary[6].push_back("NY");
  optionsThatVary[6].push_back("NZ");

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
          bw(allopt[iOpt], optionsThatVary[iOpt], subComm);
        } else if(benchmarks[iBench] == "dot") {
          dot(allopt[iOpt], optionsThatVary[iOpt], subComm);
        } else if(benchmarks[iBench] == "allreduce") {
          allred(allopt[iOpt], optionsThatVary[iOpt], subComm);
        } else if(benchmarks[iBench] == "ogs") {
          gs(allopt[iOpt], optionsThatVary[iOpt], subComm, true, false);
        } else if(benchmarks[iBench] == "pingpong") {
          gs(allopt[iOpt], optionsThatVary[iOpt], subComm, false, true);
        } else if(benchmarks[iBench] == "axhelm") {
          axhelm(allopt[iOpt], optionsThatVary[iOpt], subComm);
        } else if(benchmarks[iBench] == "nekbone") {
          nekBone(allopt[iOpt], optionsThatVary[iOpt], subComm);
        }
      }

      MPI_Barrier(comm);

    }

  }
  
}
