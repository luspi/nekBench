extern "C" {
int pingPongMulti(int pairs, int useDevice, occa::device device, MPI_Comm comm, bool driverModus, setupAide opt, std::vector<std::string> optionsForFilename);
int pingPongSingle(int useDevice, occa::device device, MPI_Comm comm, bool driverModus, setupAide opt, std::vector<std::string> optionsForFilename);
}
