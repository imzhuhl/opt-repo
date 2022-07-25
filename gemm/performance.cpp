/**
 * Performance of my implementation
 * */
#include <iostream>
#include <chrono>
#include "utils.hpp"

constexpr int MIN_SIZE = 2048;
constexpr int STRIDE = 256;
constexpr int MAX_SIZE = 2048;

int my_impl(int M, int K, int N, float *A, float *B, float *C);

int main() {
    printf("size, GFLOP/S, ms\n");
    for (int cur_size = MIN_SIZE; cur_size <= MAX_SIZE; cur_size += STRIDE) {
        int M = cur_size;
        int K = cur_size;
        int N = cur_size;

        double gflops = 2.0 * M * N * K * 1.0e-09;
        double best_time = std::numeric_limits<double>::infinity();

        float *A = (float *)malloc(M * K * sizeof(float));
        float *B = (float *)malloc(K * N * sizeof(float));
        float *C = (float *)malloc(M * N * sizeof(float));
        float *myc = (float *)malloc(M * N * sizeof(float));

        fill_array(A, M * K, InitVecFlag::RandonValue);
        fill_array(B, K * N, InitVecFlag::RandonValue);

        for (int rep = 0; rep < 4; rep++) {
            copy_array(C, myc, M * N);

            auto st = std::chrono::steady_clock::now();
            my_impl(M, K, N, A, B, myc);
            auto et = std::chrono::steady_clock::now();
            std::chrono::duration<double, std::milli> elapsed = et - st;
            double time_ms = elapsed.count();
            if (best_time > time_ms) {
                best_time = time_ms;
            }
        }
        printf("%d, %.3lf, %.2lf\n", cur_size, gflops / (best_time * 1e-3), best_time);
    }

    return 0;
}
