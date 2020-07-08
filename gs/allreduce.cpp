#include <cstdlib>
#include <occa.hpp>
#include <chrono>
#include <mpi.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cublas_v2.h>
#include "allreduce.h"

extern "C" { // Begin C Linkage
int allReduceTest(int useDevice, occa::device device) {

    int max_message_size = 1e6;
    int count = 100;

    int mpiRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

    time_t rawtime;
    struct tm *timeinfo;
    char buffer[80];
    time(&rawtime);
    timeinfo = localtime(&rawtime);

    { // cuda

        FILE *fp;

        if(mpiRank == 0) {
            printf("Starting all reduce measurement on device (cuda-aware MPI)... ");
            fflush(stdout);

            strftime(buffer, 80, "allreduce_device_%Y_%m_%d_%R.txt", timeinfo);
            fp = fopen(buffer, "w");
            fprintf(fp, "%-10s %-10s %-10s\n", "words", "loops", "timing");
            fflush(fp);
        }

        for(int size = 1, multiplier = 1; size < max_message_size; multiplier*=(size%(multiplier*10)==0 ? 10 : 1), size += multiplier) {

            double *h_send_buf = (double*) malloc(size*sizeof(double));
            for(int i = 0; i < size; ++i)
                h_send_buf[i] = i%100;
            double *d_send_buf, *d_recv_buf;
            cudaMalloc((void**)&d_send_buf, size*sizeof(double));
            cudaMalloc((void**)&d_recv_buf, size*sizeof(double));
            cudaMemset(d_recv_buf, 0, size*sizeof(double));
            cudaMemcpy(d_send_buf, h_send_buf, size*sizeof(double), cudaMemcpyHostToDevice);

            MPI_Barrier(MPI_COMM_WORLD);
            auto t_start = std::chrono::steady_clock::now();

            for(int i = 0; i < count; ++i)
                MPI_Allreduce(d_send_buf, d_recv_buf, size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            MPI_Barrier(MPI_COMM_WORLD);
            auto t_end = std::chrono::steady_clock::now();

            if(mpiRank == 0) {

                double timing = std::chrono::duration<double, std::milli>(t_end-t_start).count();

                fprintf(fp, "%-10d %-10d %-10f\n", size, count, timing/(double)count);

            }

            free(h_send_buf);
            cudaFree(d_send_buf);
            cudaFree(d_recv_buf);

        }

        if(mpiRank == 0) {
            fflush(fp);
            printf("done!\n");
            fflush(stdout);
        }

    }

    { // simple c loop for measuring

        FILE *fp;

        if(mpiRank == 0) {

            printf("Starting all reduce measurement on host (simple loop)... ");
            fflush(stdout);

            strftime(buffer, 80, "allreduce_host_%Y_%m_%d_%R.txt", timeinfo);
            fp = fopen(buffer, "w");
            fprintf(fp, "%-10s %-10s %-10s\n", "words", "loops", "timing");
            fflush(fp);

        }

        for(int size = 1, multiplier = 1; size < max_message_size; multiplier*=(size%(multiplier*10)==0 ? 10 : 1), size += multiplier) {

            double *send_buf = (double*) malloc(size*sizeof(double));
            double *recv_buf = (double*) malloc(size*sizeof(double));
            for(int i = 0; i < size; ++i)
                send_buf[i] = i%100;
            memset(recv_buf, 0, size*sizeof(double));

            MPI_Barrier(MPI_COMM_WORLD);
            auto t_start = std::chrono::steady_clock::now();

            for(int i = 0; i < count; ++i)
                MPI_Allreduce(send_buf, recv_buf, size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            MPI_Barrier(MPI_COMM_WORLD);
            auto t_end = std::chrono::steady_clock::now();

            if(mpiRank == 0) {

                double timing = std::chrono::duration<double, std::milli>(t_end-t_start).count();

                fprintf(fp, "%-10d %-10d %-10f\n", size, count, timing/(double)count);

            }

            free(send_buf);

        }

        if(mpiRank == 0) {
            fflush(fp);
            printf("done!\n");
        }

    }

    return EXIT_SUCCESS;

}

} // end C Linkage
