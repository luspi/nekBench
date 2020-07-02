#include <cstdlib>
#include <occa.hpp>
#include <chrono>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cublas_v2.h>

extern "C" { // Begin C Linkage
int vectorSumTest(int useDevice, occa::device device) {

    int max_message_size = 8e6;
    int count = 100;

    time_t rawtime;
    struct tm *timeinfo;
    char buffer[80];
    time(&rawtime);
    timeinfo = localtime(&rawtime);

    if(useDevice) {

        printf("Starting vector sum measurement using cuBlas... ");
        fflush(stdout);

        strftime(buffer,80,"vectorsum_cublas_%Y_%m_%d_%R.txt", timeinfo);
        FILE *fp = fopen(buffer, "w");
        fprintf(fp, "%-10s %-10s %-10s\n", "words", "loops", "timing");
        fflush(fp);

        cublasHandle_t cublas;

        cublasStatus_t stat = cublasCreate(&cublas);
        if(stat != CUBLAS_STATUS_SUCCESS)
            printf("CUBLAS initialization failed!\n");

        for(int size = 1, multiplier = 1; size < max_message_size; multiplier*=(size%(multiplier*10)==0 ? 10 : 1), size += multiplier) {

            double *d_buf;
            cudaMalloc((void**)&d_buf, size*sizeof(double));
            cudaMemset(d_buf, 0, size*sizeof(double));

            auto t_start = std::chrono::steady_clock::now();

            for(int c = 0; c < count; ++c) {
                double result = 0;
                cublasDasum(cublas, size, d_buf, 1, &result);
                if(result != 0) {
                    printf("device array sum failed: %d\n", result);
                    fflush(stdout);
                }
            }

            auto t_end = std::chrono::steady_clock::now();

            double timing = std::chrono::duration<double, std::milli>(t_end-t_start).count();

            fprintf(fp, "%-10d %-10d %-10f\n", size, count, timing/(double)count);

            cudaFree(&d_buf);

        }

        fflush(fp);
        printf("done!\n");

    }

    { // simple c loop for measuring

        printf("Starting vector sum measurement on host (simple loop)... ");
        fflush(stdout);

        strftime(buffer,80,"vectorsum_host_%Y_%m_%d_%R.txt", timeinfo);
        FILE *fp = fopen(buffer, "w");
        fprintf(fp, "%-10s %-10s %-10s\n", "words", "loops", "timing");
        fflush(fp);

        for(int size = 1, multiplier = 1; size < max_message_size; multiplier*=(size%(multiplier*10)==0 ? 10 : 1), size += multiplier) {

            double *h_buf = (double*) malloc(size*sizeof(double));
            memset(h_buf, 0, size*sizeof(double));

            auto t_start = std::chrono::steady_clock::now();

            for(int c = 0; c < count; ++c) {
                double result = 0;
                for(int i = 0; i < size; ++i)
                    result += fabs(h_buf[i]);
                if(result != 0) {
                    printf("host array sum failed: %d\n", result);
                    fflush(stdout);
                }

            }

            auto t_end = std::chrono::steady_clock::now();

            double timing = std::chrono::duration<double, std::milli>(t_end-t_start).count();

            fprintf(fp, "%-10d %-10d %-10f\n", size, count, timing/(double)count);

            free(h_buf);

        }

        fflush(fp);
        printf("done!\n");

    }

    return EXIT_SUCCESS;

}

} // end C Linkage
