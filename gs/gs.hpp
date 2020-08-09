#include "setupAide.hpp"

void gs(setupAide &options, MPI_Comm mpiComm, bool testOgsModes, bool testPingPong);

void ogs(setupAide &options, MPI_Comm mpiComm) { gs(options, mpiComm, true, false); }
void pingpong(setupAide &options, MPI_Comm mpiComm) { gs(options, mpiComm, false, true); }
