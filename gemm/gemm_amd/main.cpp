#include "utils.hpp"

#include <chrono>
#include <iostream>
#include <vector>

int my_impl(int M, int K, int N, float *A, int lda, float *B, int ldb, float *C, int ldc);

int native_c(int M, int K, int N, float *A, int lda, float *B, int ldb, float *C, int ldc) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float tmp = 0.0;
            for (int p = 0; p < K; p++) {
                tmp += A[lda * i + p] * B[ldb * p + j];
            }
            C[ldc * i + j] += tmp;
        }
    }
    return 0;
}


int main() {
#ifdef DEBUG
    constexpr int SIZE = 16;
#else
    constexpr int SIZE = 1024;
#endif
    constexpr int M = SIZE;
    constexpr int K = SIZE;
    constexpr int N = SIZE;

    // row major
    int lda = K;
    int ldb = N;
    int ldc = N;

    double gflops = 1.0 * M * N * K * 1.0e-09;

    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N, 0);

    std::vector<float> refc(C);
    std::vector<float> myc(C);

    fill_array(A.data(), M * K, InitVecFlag::RandonValue);
    fill_array(B.data(), K * N, InitVecFlag::RandonValue);

#ifdef DEBUG
    printf("Matrix A:\n");
    display_matrix(A.data(), A.size(), lda);
    printf("Matrix B:\n");
    display_matrix(B.data(), B.size(), lda);
#endif

    // perf usage
    // while (true) {
    //     myc.assign(C.begin(), C.end());
    //     my_impl(M, K, N, A.data(), lda, B.data(), ldb, myc.data(), ldc);
    // }

    for (int rep = 0; rep < 4; rep++) {
        myc.assign(C.begin(), C.end());
        
        auto start = std::chrono::steady_clock::now();        
        my_impl(M, K, N, A.data(), lda, B.data(), ldb, myc.data(), ldc);
        auto end = std::chrono::steady_clock::now();

        std::chrono::duration<double, std::milli> elpased = end - start;
        double time = elpased.count() * 1.0e-3;
        printf("%.3lf GFLOP/S, %.2lf ms\n", gflops / time, elpased.count());
    }


    native_c(M, K, N, A.data(), lda, B.data(), ldb, refc.data(), ldc);
    compare_array(refc.data(), myc.data(), M * N);

#ifdef DEBUG
    printf("Matrix refc:\n");
    display_matrix(refc.data(), refc.size(), ldc);
    printf("Matrix myc:\n");
    display_matrix(myc.data(), myc.size(), ldc);
#endif 

    return 0;
}
