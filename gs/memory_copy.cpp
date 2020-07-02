#include <cstdlib>
#include <occa.hpp>
#include <chrono>

extern "C" { // Begin C Linkage
int deviceMemcpyTest(occa::device device) {

    int min_message_size = 1;
    int max_message_size = 1e7;
    int count = 100;

    time_t rawtime;
    struct tm *timeinfo;
    char buffer[80];
    time(&rawtime);
    timeinfo = localtime(&rawtime);

    {

        printf("Starting device-to-host memcpy measurement... ");
        fflush(stdout);

        strftime(buffer,80,"memcpy_devicetohost_%Y_%m_%d_%R.txt", timeinfo);
        FILE *fp = fopen(buffer, "w");
        fprintf(fp, "%-10s %-10s\n", "words", "timing");

        for(int size = min_message_size; size <= max_message_size; size=((int)(size*1.1) > size ? (int)(size*1.1) : size+1)) {

            occa::memory d_buf = device.malloc(size*sizeof(double));
            double *h_buf = new double[size];

            // warmup
            for(int i = 0; i < 10; ++i)
                d_buf.copyTo(h_buf, size*sizeof(double), 0);

            auto t_start = std::chrono::steady_clock::now();
            for(int i = 0; i < count; ++i)
                d_buf.copyTo(h_buf, size*sizeof(double), 0);
            auto t_end = std::chrono::steady_clock::now();

            double timing = std::chrono::duration<double, std::milli>(t_end-t_start).count();

            fprintf(fp, "%-10d %-10f\n", size, timing/(double)count);

            delete[] h_buf;

        }

        printf("done.\n");

    }

    {

        printf("Starting host-to-device memcpy measurement... ");
        fflush(stdout);

        strftime(buffer,80,"memcpy_hosttodevice_%Y_%m_%d_%R.txt", timeinfo);
        FILE *fp = fopen(buffer, "w");
        fprintf(fp, "%-10s %-10s\n", "words", "timing");

        for(int size = min_message_size; size <= max_message_size; size=((int)(size*1.1) > size ? (int)(size*1.1) : size+1)) {

            occa::memory d_buf = device.malloc(size*sizeof(double));
            double *h_buf = new double[size]{};

            // warmup
            for(int i = 0; i < 10; ++i)
                d_buf.copyFrom(h_buf, size*sizeof(double), 0);

            auto t_start = std::chrono::steady_clock::now();
            for(int i = 0; i < count; ++i)
                d_buf.copyFrom(h_buf, size*sizeof(double), 0);
            auto t_end = std::chrono::steady_clock::now();

            double timing = std::chrono::duration<double, std::milli>(t_end-t_start).count();

            fprintf(fp, "%-10d %-10f\n", size, timing/(double)count);

            delete[] h_buf;

        }

        printf("done.\n");

    }

    return EXIT_SUCCESS;

}
} // end C Linkage
