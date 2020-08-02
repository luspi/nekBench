extern "C" {
int pingPongMulti(int pairs, int useDevice, occa::device device, MPI_Comm comm, bool driverModus);
int pingPongSingle(int useDevice, occa::device device, MPI_Comm comm, bool driverModus);
}
