#include <iostream>
#include <chrono>
#include <arm_neon.h>
#include "utils.hpp"

constexpr int MIN_SIZE = 2048;
constexpr int STRIDE = 256;
constexpr int MAX_SIZE = 2048;

float bf16_to_fp32(bfloat16 src) {
    float rst = 0;
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    *(reinterpret_cast<bfloat16 *>(&rst)) = src;
#else
    *(reinterpret_cast<bfloat16 *>(&rst)+1) = src;
#endif
    return rst;
}

int native_c(int M, int K, int N, bfloat16 *A, bfloat16 *B, float *C) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float tmp = 0.0;
            for (int p = 0; p < K; p++) {
                tmp += bf16_to_fp32(A[M * p + i]) * bf16_to_fp32(B[K * j + p]);
            }
            C[M * j + i] += tmp;
        }
    }
    return 0;
}


void packA(int M, int N, bfloat16 *A, int lda, bfloat16 *sa) {
    for (int i = 0; i < M; i += 4) {
        for (int j = 0; j < N; j += 4) {
            sa[0] = A[j * lda + i];
            sa[1] = A[(j + 1) * lda + i];
            sa[2] = A[(j + 2) * lda + i];
            sa[3] = A[(j + 3) * lda + i];
            sa[4] = A[j * lda + i + 1];
            sa[5] = A[(j + 1) * lda + i + 1];
            sa[6] = A[(j + 2) * lda + i + 1];
            sa[7] = A[(j + 3) * lda + i + 1];
            sa[8] = A[j * lda + i + 2];
            sa[9] = A[(j + 1) * lda + i + 2];
            sa[10] = A[(j + 2) * lda + i + 2];
            sa[11] = A[(j + 3) * lda + i + 2];
            sa[12] = A[j * lda + i + 3];
            sa[13] = A[(j + 1) * lda + i + 3];
            sa[14] = A[(j + 2) * lda + i + 3];
            sa[15] = A[(j + 3) * lda + i + 3];
            sa += 16; 
        }
    }
}

void packB(int M, int N, bfloat16 *B, int ldb, bfloat16 *sb) {
    for (int j = 0; j < N; j += 4) {
        for (int i = 0; i < M; i += 4) {
            sb[0] = B[j * ldb + i];
            sb[1] = B[j * ldb + i + 1];
            sb[2] = B[j * ldb + i + 2];
            sb[3] = B[j * ldb + i + 3];
            sb[4] = B[(j + 1) * ldb + i];
            sb[5] = B[(j + 1) * ldb + i + 1];
            sb[6] = B[(j + 1) * ldb + i + 2];
            sb[7] = B[(j + 1) * ldb + i + 3];
            sb[8] = B[(j + 2) * ldb + i];
            sb[9] = B[(j + 2) * ldb + i + 1];
            sb[10] = B[(j + 2) * ldb + i + 2];
            sb[11] = B[(j + 2) * ldb + i + 3];
            sb[12] = B[(j + 3) * ldb + i];
            sb[13] = B[(j + 3) * ldb + i + 1];
            sb[14] = B[(j + 3) * ldb + i + 2];
            sb[15] = B[(j + 3) * ldb + i + 3];

            sb += 16;
        }
    }

#if 0
    sb = sb - M * N;
    for (int i = 0; i < M * N; i++) {
        printf(" %lf\n", bf16_to_fp32(sb[i]));
    }
#endif
}

void kernel(int M, int K, int N, bfloat16 *sa, bfloat16 *sb, float *sc) {
    bfloat16x8_t ma0, ma1, mb0, mb1;
    bfloat16 *ptr_a, *ptr_b;
    float32x4_t mc00, mc10, mc01, mc11;
    float32x4_t vc0, vc1, vc2, vc3;
    
    float *ptr_c = sc;
    for (int i = 0; i < M; i += 4) {
        for (int j = 0; j < N; j += 4) {
            ptr_a = &sa[i * K];
            ptr_b = &sb[j * K];
            mc00 = vdupq_n_f32(0);
            mc01 = vdupq_n_f32(0);
            mc10 = vdupq_n_f32(0);
            mc11 = vdupq_n_f32(0);
            for (int p = 0; p < K; p += 4) {
                ma0 = vld1q_bf16(reinterpret_cast<bfloat16_t *>(ptr_a));
                ma1 = vld1q_bf16(reinterpret_cast<bfloat16_t *>(ptr_a + 8));
                mb0 = vld1q_bf16(reinterpret_cast<bfloat16_t *>(ptr_b));
                mb1 = vld1q_bf16(reinterpret_cast<bfloat16_t *>(ptr_b + 8));
                mc00 = vbfmmlaq_f32(mc00, ma0, mb0);
                mc01 = vbfmmlaq_f32(mc01, ma0, mb1);
                mc10 = vbfmmlaq_f32(mc10, ma1, mb0);
                mc11 = vbfmmlaq_f32(mc11, ma1, mb1);
                ptr_a += 16;
                ptr_b += 16;
            }
            vc0 = vuzp1q_f32(mc00, mc10);
            vc1 = vuzp2q_f32(mc00, mc10);
            vc2 = vuzp1q_f32(mc01, mc11);
            vc3 = vuzp2q_f32(mc01, mc11);

            vst1q_f32(ptr_c, vc0);
            vst1q_f32(ptr_c + 4, vc1);
            vst1q_f32(ptr_c + 8, vc2);
            vst1q_f32(ptr_c + 12, vc3);
            ptr_c += 16;
        }
    }
}

void merge(int M, int N, float *sc, float *C, int ldc) {
    float32x4_t src_c, dst_c;
    float *ptr_c = sc;
    for (int i = 0; i < M; i += 4) {
        for (int j = 0; j < N; j++) {
            src_c = vld1q_f32(&C[j * ldc + i]);
            dst_c = vld1q_f32(ptr_c);
            ptr_c += 4;
            dst_c = vaddq_f32(dst_c, src_c);
            vst1q_f32(&C[j * ldc + i], dst_c); 
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

    float *tmpc = (float *) malloc(BLOCK_M * BLOCK_N * sizeof(float));

    for (int i = 0; i < M; i += BLOCK_M) {
        for (int p = 0; p < K; p += BLOCK_K) {
            packA(BLOCK_M, BLOCK_K, A + p * M + i, M, sa);
            for (int j = 0; j < N; j += BLOCK_N) {
                packB(BLOCK_K, BLOCK_N, B + j * K + p, K, sb);
                // kernel(BLOCK_M, BLOCK_K, BLOCK_N, sa, sb, C + j * M + i, M);
                kernel(BLOCK_M, BLOCK_K, BLOCK_N, sa, sb, tmpc);
                merge(BLOCK_M, BLOCK_N, tmpc, C + j * M + i, M);
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

#ifdef DEBUG
    fill_array(FA, M * K, InitVecFlag::IncreaseByOne);
    fill_array(FB, K * N, InitVecFlag::IncreaseByOne);
#else
    fill_array(FA, M * K, InitVecFlag::RandonValue);
    fill_array(FB, K * N, InitVecFlag::RandonValue);
#endif
    fill_array(refc, K * N, InitVecFlag::Zero);
    fill_array(myc, K * N, InitVecFlag::Zero);

    array_fp32_to_bf16(FA, A, M * K);
    array_fp32_to_bf16(FB, B, K * N);

#ifdef DEBUG
    printf("Matrix A:\n");
    display_matrix(FA, M, K, ArrangeMode::ColMajor);
    printf("Matrix B:\n");
    display_matrix(FB, K, N, ArrangeMode::ColMajor);
#endif

    native_c(M, K, N, A, B, refc);
    my_impl(M, K, N, A, B, myc);
    compare_array(refc, myc, M * N);

#ifdef DEBUG
    printf("Matrix refc:\n");
    display_matrix(refc, M, N, ArrangeMode::ColMajor);
    printf("Matrix myc:\n");
    display_matrix(myc, M, N, ArrangeMode::ColMajor);
#endif 

    free(FA);
    free(FB);
    free(A);
    free(B);
    free(refc);
    free(myc);
    return 0;
}

/**
 * Column-major matrix, because it is easy for sbgemm implementation
 */ 
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
