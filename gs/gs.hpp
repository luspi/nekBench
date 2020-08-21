#include "setupAide.hpp"

void gs(setupAide &options, std::vector<std::string> optionsForFilename, MPI_Comm mpiComm, bool testOgsModes, bool testPingPong);

void ogs(setupAide &options, std::vector<std::string> optionsForFilename, MPI_Comm mpiComm) { gs(options, optionsForFilename, mpiComm, true, false); }
void pingpong(setupAide &options, std::vector<std::string> optionsForFilename, MPI_Comm mpiComm) { gs(options, optionsForFilename, mpiComm, false, true); }
