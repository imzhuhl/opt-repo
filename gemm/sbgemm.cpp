#include <iostream>
#include <chrono>
#include <arm_neon.h>
#include "utils.hpp"

constexpr int MIN_SIZE = 2048;
constexpr int STRIDE = 256;
constexpr int MAX_SIZE = 2048;

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


void packA(int M, int N, bfloat16 *A, int lda, bfloat16 *sa) {
    for (int i = 0; i < M; i += 4) {
        for (int j = 0; j < N; j += 4) {
            sa[0] = A[i * lda + j];
            sa[1] = A[i * lda + j + 1];
            sa[2] = A[i * lda + j + 2];
            sa[3] = A[i * lda + j + 3];
            sa[4] = A[(i + 1) * lda + j];
            sa[5] = A[(i + 1) * lda + j + 1];
            sa[6] = A[(i + 1) * lda + j + 2];
            sa[7] = A[(i + 1) * lda + j + 3];
            sa[8] = A[(i + 2) * lda + j];
            sa[9] = A[(i + 2) * lda + j + 1];
            sa[10] = A[(i + 2) * lda + j + 2];
            sa[11] = A[(i + 2) * lda + j + 3];
            sa[12] = A[(i + 3) * lda + j];
            sa[13] = A[(i + 3) * lda + j + 1];
            sa[14] = A[(i + 3) * lda + j + 2];
            sa[15] = A[(i + 3) * lda + j + 3];

            sa += 16; 
        }
    }
}

void packB(int M, int N, bfloat16 *B, int ldb, bfloat16 *sb) {
    for (int j = 0; j < N; j += 2) {
        for (int i = 0; i < M; i += 4) {
            sb[0] = B[i * ldb + j];
            sb[1] = B[(i + 1) * ldb + j];
            sb[2] = B[(i + 2) * ldb + j];
            sb[3] = B[(i + 3) * ldb + j];
            sb[4] = B[i * ldb + j + 1];
            sb[5] = B[(i + 1) * ldb + j + 1];
            sb[6] = B[(i + 2) * ldb + j + 1];
            sb[7] = B[(i + 3) * ldb + j + 1];
            sb += 8;
        }
    }
}

void kernel(int M, int K, int N, bfloat16 *sa, bfloat16 *sb, float *C) {
    bfloat16x8_t va0, va1, vb;
    float32x4_t vc0, vc1;
    for (int i = 0; i < M; i += 4) {
        for (int j = 0; j < N; j += 2) {
            vc0 = vdupq_n_f32(0);
            vc1 = vdupq_n_f32(0);
            for (int p = 0; p < K / 4; p++) {
                va0 = vld1q_bf16(reinterpret_cast<bfloat16_t *>(&sa[i * K + p * 16]));
                va1 = vld1q_bf16(reinterpret_cast<bfloat16_t *>(&sa[i * K + p * 16 + 8]));
                vb = vld1q_bf16(reinterpret_cast<bfloat16_t *>(&sb[j * K + p * 8]));
                vc0 = vbfmmlaq_f32(vc0, va0, vb);
                vc1 = vbfmmlaq_f32(vc1, va1, vb);
            }
        }
    }

}

int my_impl(int M, int K, int N, bfloat16 *A, bfloat16 *B, float *C) {
#ifdef DEBUG
    constexpr int BLOCK_M = 8;
    constexpr int BLOCK_K = 8;
    constexpr int BLOCK_N = 8;
#else
    constexpr int BLOCK_M = 256;
    constexpr int BLOCK_K = 256;
    constexpr int BLOCK_N = 256;
#endif

    bfloat16 *buffer = (bfloat16 *) malloc ((BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * sizeof(bfloat16));

    bfloat16 *sa = buffer;
    bfloat16 *sb = buffer + BLOCK_M * BLOCK_K;

    float *tmpc = (float *) malloc(4 * 2 * sizeof(float));

    for (int i = 0; i < M; i += BLOCK_M) {
        for (int p = 0; p < K; p += BLOCK_K) {
            packA(BLOCK_M, BLOCK_K, A + i * K + p, K, sa);
            for (int j = 0; j < N; j += BLOCK_N) {
                packB(BLOCK_K, BLOCK_N, B + p * N + j, N, sb);
                // kernel(BLOCK_M, BLOCK_K, BLOCK_N, sa, sb, C + i * N + j, N);
                kernel(BLOCK_M, BLOCK_K, BLOCK_N, sa, sb, tmpc);
            }
        }
    }


    free(buffer);
    free(tmpc);
    return 0;
}


int check() {
#ifdef DEBUG
    constexpr int SIZE = 8;
#else
    constexpr int SIZE = 1024;
#endif

    constexpr int M = SIZE;
    constexpr int K = SIZE;
    constexpr int N = SIZE;

    float *FA = (float *)malloc(M * K * sizeof(float));
    float *FB = (float *)malloc(K * N * sizeof(float));
    bfloat16 *A = (bfloat16 *)malloc(M * K * sizeof(bfloat16));
    bfloat16 *B = (bfloat16 *)malloc(K * N * sizeof(bfloat16));
    float *refc = (float *)malloc(M * N * sizeof(float));
    float *myc = (float *)malloc(M * N * sizeof(float));

    fill_array(FA, M * K, InitVecFlag::IncreaseByOne);
    fill_array(FB, K * N, InitVecFlag::One);
    fill_array(refc, K * N, InitVecFlag::Zero);
    fill_array(myc, K * N, InitVecFlag::Zero);

    array_fp32_to_bf16(FA, A, M * K);
    array_fp32_to_bf16(FB, B, K * N);

#ifdef DEBUG
    printf("Matrix A:\n");
    display_matrix(FA, M, K);
    printf("Matrix B:\n");
    display_matrix(FB, K, N);
#endif

    native_c(M, K, N, FA, FB, refc);
    my_impl(M, K, N, A, B, myc);
    compare_array(refc, myc, M * N);

    free(FA);
    free(FB);
    free(A);
    free(B);
    free(refc);
    free(myc);
    return 0;
}

int main() {
#ifdef CHECK
    check();
#else
    printf("size, GFLOP/S, ms\n");
    for (int cur_size = MIN_SIZE; cur_size <= MAX_SIZE; cur_size += STRIDE) {
        int M = cur_size;
        int K = cur_size;
        int N = cur_size;

        double gflops = 2.0 * M * N * K * 1.0e-09;
        double best_time = std::numeric_limits<double>::infinity();

        float *FA = (float *)malloc(M * K * sizeof(float));
        float *FB = (float *)malloc(K * N * sizeof(float));
        bfloat16 *A = (bfloat16 *)malloc(M * K * sizeof(bfloat16));
        bfloat16 *B = (bfloat16 *)malloc(K * N * sizeof(bfloat16));
        float *C = (float *)malloc(M * N * sizeof(float));
        float *myc = (float *)malloc(M * N * sizeof(float));

        fill_array(FA, M * K, InitVecFlag::RandonValue);
        fill_array(FB, K * N, InitVecFlag::RandonValue);
        array_fp32_to_bf16(FA, A, M * K);
        array_fp32_to_bf16(FB, B, K * N);
        fill_array(C, K * N, InitVecFlag::Zero);

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

        free(FA);
        free(FB);
        free(A);
        free(B);
        free(myc);
    }
#endif

}
