/**
 * check whether it is correct
 * */

#include "utils.hpp"

int my_impl(int M, int K, int N, float *A, float *B, float *C);

int native_c(int M, int K, int N, float *A, float *B, float *C) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float tmp = 0.0;
            for (int p = 0; p < K; p++) {
                tmp += A[K * i + p] * B[N * p + j];
            }
            C[N * i + j] += tmp;
        }
    }
    return 0;
}

int main() {
#ifdef DEBUG
    constexpr int SIZE = 4;
#else
    constexpr int SIZE = 1024;
#endif

    constexpr int M = SIZE;
    constexpr int K = SIZE;
    constexpr int N = SIZE;

    float *A = (float *)malloc(M * K * sizeof(float));
    float *B = (float *)malloc(K * N * sizeof(float));
    float *refc = (float *)malloc(M * N * sizeof(float));
    float *myc = (float *)malloc(M * N * sizeof(float));

    fill_array(A, M * K, InitVecFlag::IncreaseByOne);
    fill_array(B, K * N, InitVecFlag::One);
    fill_array(refc, K * N, InitVecFlag::Zero);
    fill_array(myc, K * N, InitVecFlag::Zero);

#ifdef DEBUG
    printf("Matrix A:\n");
    display_matrix(A, M, K);
    printf("Matrix B:\n");
    display_matrix(B, K, N);
#endif

    native_c(M, K, N, A, B, refc);
    my_impl(M, K, N, A, B, myc);
    compare_array(refc, myc, M * N);

#ifdef DEBUG
    printf("Matrix refc:\n");
    display_matrix(refc, M, N);
    printf("Matrix myc:\n");
    display_matrix(myc, M, N);
#endif

    return 0;
}
