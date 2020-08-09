#include <string>
#include <iostream>
#include "../core/setCompilerFlags.hpp"
#include "../axhelm/axhelm.hpp"
#include "../bw/bw.hpp"
#include "../dot/dot.hpp"
#include "../gs/gs.hpp"
#include "../nekBone/nekBone.hpp"
#include "../core/parReader.hpp"

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

void driver(std::string inifile, MPI_Comm comm) {

  int mpiRank, mpiSize;
  MPI_Comm_rank(comm, &mpiRank);
  MPI_Comm_size(comm, &mpiSize);

  ParRead parReader(inifile);

  std::string benchmarks[6] = {"bw", "dot", "ogs", "pingpong", "axhelm", "nekbone"};
  for(int iBench = 0; iBench < 6; ++iBench) {

    std::vector<setupAide> allopt = parReader.getOptions(benchmarks[iBench]);

    for(int iOpt = 0; iOpt < allopt.size(); ++iOpt) {

      bool enabled = (allopt[iOpt].compareArgs("ENABLED", "TRUE"));
      if(!enabled) break;

      int howManyMpiRanks;
      if(allopt[iOpt].compareArgs("MPI", "MAX"))
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
        if(iBench == 0) {
          bw(allopt[iOpt]);
        } else if(iBench == 1) {
          dot(allopt[iOpt], subComm);
        } else if(iBench == 2) {
          gs(allopt[iOpt], subComm, true, false);
        } else if(iBench == 3) {
          gs(allopt[iOpt], subComm, false, true);
        } else if(iBench == 4) {
          axhelm(allopt[iOpt]);
        } else if(iBench == 5) {
          nekBone(allopt[iOpt]);
        }
      }

      MPI_Barrier(comm);

    }

  }
  
}
