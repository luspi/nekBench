#include <cstdlib>
#include <occa.hpp>
#include <chrono>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cublas_v2.h>

extern "C" { // Begin C Linkage
int vectorDotProduct(int useDevice, occa::device device) {

    int max_message_size = 8e6;
    int count = 100;

    time_t rawtime;
    struct tm *timeinfo;
    char buffer[80];
    time(&rawtime);
    timeinfo = localtime(&rawtime);

    if(useDevice) {

        printf("Starting dot product measurement using cuBlas... ");
        fflush(stdout);

        strftime(buffer,80,"dotproduct_cublas_%Y_%m_%d_%R.txt", timeinfo);
        FILE *fp = fopen(buffer, "w");
        fprintf(fp, "%-10s %-10s %-10s\n", "words", "loops", "timing");
        fflush(fp);

        cublasHandle_t cublas;

        cublasStatus_t stat = cublasCreate(&cublas);
        if(stat != CUBLAS_STATUS_SUCCESS)
            printf("CUBLAS initialization failed!\n");

        for(int size = 1, multiplier = 1; size < max_message_size; multiplier*=(size%(multiplier*10)==0 ? 10 : 1), size += multiplier) {

            double *d_buf1, *d_buf2;
            cudaMalloc((void**)&d_buf1, size*sizeof(double));
            cudaMalloc((void**)&d_buf2, size*sizeof(double));
            cudaMemset(d_buf1, 0, size*sizeof(double));
            cudaMemset(d_buf2, 0, size*sizeof(double));

            auto t_start = std::chrono::steady_clock::now();

            for(int c = 0; c < count; ++c) {
                double result = 0;
                cublasDdot(cublas, size, d_buf1, 1, d_buf2, 1, &result);
                if(result != 0) {
                    printf("device dot product failed: %d\n", result);
                    fflush(stdout);
                }
            }

            auto t_end = std::chrono::steady_clock::now();

            double timing = std::chrono::duration<double, std::milli>(t_end-t_start).count();

            fprintf(fp, "%-10d %-10d %-10f\n", size, count, timing/(double)count);

            cudaFree(&d_buf1);
            cudaFree(&d_buf2);

        }

        fflush(fp);
        printf("done!\n");

    }

    { // simple c loop for measuring

        printf("Starting vector sum measurement on host (simple loop)... ");
        fflush(stdout);

        strftime(buffer,80,"dotproduct_host_%Y_%m_%d_%R.txt", timeinfo);
        FILE *fp = fopen(buffer, "w");
        fprintf(fp, "%-10s %-10s %-10s\n", "words", "loops", "timing");
        fflush(fp);

        for(int size = 1, multiplier = 1; size < max_message_size; multiplier*=(size%(multiplier*10)==0 ? 10 : 1), size += multiplier) {

            double *h_buf1 = (double*) malloc(size*sizeof(double));
            double *h_buf2 = (double*) malloc(size*sizeof(double));
            memset(h_buf1, 0, size*sizeof(double));
            memset(h_buf2, 0, size*sizeof(double));

            auto t_start = std::chrono::steady_clock::now();

            for(int c = 0; c < count; ++c) {
                double result = 0;
                for(int i = 0; i < size; ++i)
                    result += h_buf1[i]*h_buf2[i];
                if(result != 0) {
                    printf("host dot product failed: %d\n", result);
                    fflush(stdout);
                }

            }

            auto t_end = std::chrono::steady_clock::now();

            double timing = std::chrono::duration<double, std::milli>(t_end-t_start).count();

            fprintf(fp, "%-10d %-10d %-10f\n", size, count, timing/(double)count);

            free(h_buf1);
            free(h_buf2);

        }

        fflush(fp);
        printf("done!\n");

    }

    return EXIT_SUCCESS;

}

} // end C Linkage
