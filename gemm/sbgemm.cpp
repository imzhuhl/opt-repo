#include <arm_neon.h>
#include <chrono>
#include <iostream>
#include "utils.hpp"

constexpr int MIN_SIZE = 256;
constexpr int STRIDE = 256;
constexpr int MAX_SIZE = 2048;

float bf16_to_fp32(bfloat16 src) {
    float rst = 0;
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    *(reinterpret_cast<bfloat16 *>(&rst)) = src;
#else
    *(reinterpret_cast<bfloat16 *>(&rst) + 1) = src;
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

void packA_8(int M, int N, bfloat16 *A, int lda, bfloat16 *sa) {
    for (int i = 0; i < M; i += 8) {
        for (int j = 0; j < N; j += 4) {
            for (int ii = 0; ii < 8; ii++) {
                for (int jj = 0; jj < 4; jj++) {
                    sa[ii * 4 + jj] = A[(j + jj) * lda + i + ii];
                }
            }

            sa += 32;
        }
    }
}

void packA_12(int M, int N, bfloat16 *A, int lda, bfloat16 *sa) {
    for (int i = 0; i < M; i += 12) {
        for (int j = 0; j < N; j += 4) {
            for (int ii = 0; ii < 12; ii++) {
                for (int jj = 0; jj < 4; jj++) {
                    sa[ii * 4 + jj] = A[(j + jj) * lda + i + ii];
                }
            }

            sa += 48;
        }
    }
}

void packB_4(int M, int N, bfloat16 *B, int ldb, bfloat16 *sb) {
    uint16x4_t v0;
    for (int j = 0; j < N; j += 4) {
        for (int i = 0; i < M; i += 4) {
            for (int jj = 0; jj < 4; jj++) {
                v0 = vld1_u16(reinterpret_cast<uint16_t *>(&B[(j + jj) * ldb + i]));
                vst1_u16(reinterpret_cast<uint16_t *>(sb + jj * 4), v0);
            }
            sb += 16;
        }
    }
}

void packB_8(int M, int N, bfloat16 *B, int ldb, bfloat16 *sb) {
    uint16x4_t v0;
    for (int j = 0; j < N; j += 8) {
        for (int i = 0; i < M; i += 4) {
            for (int jj = 0; jj < 8; jj++) {
                v0 = vld1_u16(reinterpret_cast<uint16_t *>(&B[(j + jj) * ldb + i]));
                vst1_u16(reinterpret_cast<uint16_t *>(sb + jj * 4), v0);
            }
            sb += 32;
        }
    }

#if 0
    sb = sb - M * N;
    for (int i = 0; i < M * N; i++) {
        printf(" %lf\n", bf16_to_fp32(sb[i]));
    }
#endif
}

#define LOAD_C(M, N) mc##M##N = vdupq_n_f32(0);

#define MATMUL(M, N) mc##M##N = vbfmmlaq_f32(mc##M##N, ma##M, mb##N);

void bfmmla_8x8(int K, bfloat16_t *sa, bfloat16_t *sb, float *sc) {
    bfloat16x8_t ma0, ma1, ma2, ma3, ma4, ma5, mb0, mb1, mb2, mb3;
    float32x4_t mc00, mc01, mc02, mc03, 
                mc10, mc11, mc12, mc13, 
                mc20, mc21, mc22, mc23, 
                mc30, mc31, mc32, mc33;
    float32x4_t vc0, vc1, vc2, vc3, vc4, vc5, vc6, vc7,
                vc8, vc9, vc10, vc11, vc12, vc13, vc14, vc15;

    bfloat16_t *ptr_a = sa;
    bfloat16_t *ptr_b = sb;
    float *ptr_c = sc;

    LOAD_C(0, 0); LOAD_C(0, 1); LOAD_C(0, 2); LOAD_C(0, 3);
    LOAD_C(1, 0); LOAD_C(1, 1); LOAD_C(1, 2); LOAD_C(1, 3);
    LOAD_C(2, 0); LOAD_C(2, 1); LOAD_C(2, 2); LOAD_C(2, 3);
    LOAD_C(3, 0); LOAD_C(3, 1); LOAD_C(3, 2); LOAD_C(3, 3);

    for (int p = 0; p < K; p += 4) {
        ma0 = vld1q_bf16(ptr_a);
        ma1 = vld1q_bf16(ptr_a + 8);
        ma2 = vld1q_bf16(ptr_a + 16);
        ma3 = vld1q_bf16(ptr_a + 24);

        mb0 = vld1q_bf16(ptr_b);
        mb1 = vld1q_bf16(ptr_b + 8);
        mb2 = vld1q_bf16(ptr_b + 16);
        mb3 = vld1q_bf16(ptr_b + 24);

        MATMUL(0, 0); MATMUL(0, 1); MATMUL(0, 2); MATMUL(0, 3);
        MATMUL(1, 0); MATMUL(1, 1); MATMUL(1, 2); MATMUL(1, 3);
        MATMUL(2, 0); MATMUL(2, 1); MATMUL(2, 2); MATMUL(2, 3);
        MATMUL(3, 0); MATMUL(3, 1); MATMUL(3, 2); MATMUL(3, 3);

        ptr_a += 32;
        ptr_b += 32;
    }
    vc0 = vuzp1q_f32(mc00, mc10);
    vc1 = vuzp1q_f32(mc20, mc30);
    vc2 = vuzp2q_f32(mc00, mc10);
    vc3 = vuzp2q_f32(mc20, mc30);

    vc4 = vuzp1q_f32(mc01, mc11);
    vc5 = vuzp1q_f32(mc21, mc31);
    vc6 = vuzp2q_f32(mc01, mc11);
    vc7 = vuzp2q_f32(mc21, mc31);

    vc8  = vuzp1q_f32(mc02, mc12);
    vc9  = vuzp1q_f32(mc22, mc32);
    vc10 = vuzp2q_f32(mc02, mc12);
    vc11 = vuzp2q_f32(mc22, mc32);

    vc12 = vuzp1q_f32(mc03, mc13);
    vc13 = vuzp1q_f32(mc23, mc33);
    vc14 = vuzp2q_f32(mc03, mc13);
    vc15 = vuzp2q_f32(mc23, mc33);

    vst1q_f32(ptr_c, vc0);
    vst1q_f32(ptr_c + 4, vc1);
    vst1q_f32(ptr_c + 8, vc2);
    vst1q_f32(ptr_c + 12, vc3);
    vst1q_f32(ptr_c + 16, vc4);
    vst1q_f32(ptr_c + 20, vc5);
    vst1q_f32(ptr_c + 24, vc6);
    vst1q_f32(ptr_c + 28, vc7);
    vst1q_f32(ptr_c + 32, vc8);
    vst1q_f32(ptr_c + 36, vc9);
    vst1q_f32(ptr_c + 40, vc10);
    vst1q_f32(ptr_c + 44, vc11);
    vst1q_f32(ptr_c + 48, vc12);
    vst1q_f32(ptr_c + 52, vc13);
    vst1q_f32(ptr_c + 56, vc14);
    vst1q_f32(ptr_c + 60, vc15);
}

void merge_8x8(float *C, int ldc, float *sc) {
    float *output_c;
    float32x4_t src_vc, rst_vc;
    for (int j = 0; j < 8; j++) {
        output_c = C + j * ldc;
        for (int i = 0; i < 8; i += 4) {
            src_vc = vld1q_f32(sc + j * 8 + i);
            rst_vc = vld1q_f32(output_c + i);
            rst_vc = vaddq_f32(src_vc, rst_vc);
            vst1q_f32(output_c + i, rst_vc);
        }
    }
}

void merge_12x8(float *C, int ldc, float *sc) {
    float *output_c;
    float32x4_t src_vc, rst_vc;
    for (int j = 0; j < 8; j++) {
        output_c = C + j * ldc;
        for (int i = 0; i < 12; i += 4) {
            src_vc = vld1q_f32(sc + j * 8 + i);
            rst_vc = vld1q_f32(output_c + i);
            rst_vc = vaddq_f32(src_vc, rst_vc);
            vst1q_f32(output_c + i, rst_vc);
        }
    }
}

void kernel(int M, int K, int N, bfloat16 *sa, bfloat16 *sb, float *C, int ldc) {
    float *tmpc = (float *)malloc(12 * 8 * sizeof(float));

    bfloat16x8_t ma0, ma1, ma2, ma3, mb0, mb1, mb2, mb3;
    bfloat16_t *ptr_a, *ptr_b;
    float *ptr_c;

    float32x4_t mc00, mc01, mc02, mc03, mc10, mc11, mc12, mc13, mc20, mc21, mc22, mc23, mc30, mc31,
        mc32, mc33;
    float32x4_t vc0, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8, vc9, vc10, vc11, vc12, vc13, vc14,
        vc15;

    for (int i = 0; i < M; i += 8) {
        for (int j = 0; j < N; j += 8) {
            bfmmla_8x8(K, reinterpret_cast<bfloat16_t *>(&sa[i * K]),
                        reinterpret_cast<bfloat16_t *>(&sb[j * K]), tmpc);
            merge_8x8(C + j * ldc + i, ldc, tmpc);
        }
    }

    free(tmpc);
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

    bfloat16 *buffer =
        (bfloat16 *)malloc((BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * sizeof(bfloat16));

    bfloat16 *sa = buffer;
    bfloat16 *sb = buffer + BLOCK_M * BLOCK_K;

    for (int i = 0; i < M; i += BLOCK_M) {
        for (int p = 0; p < K; p += BLOCK_K) {
            packA_8(BLOCK_M, BLOCK_K, A + p * M + i, M, sa);
            for (int j = 0; j < N; j += BLOCK_N) {
                packB_8(BLOCK_K, BLOCK_N, B + j * K + p, K, sb);
                kernel(BLOCK_M, BLOCK_K, BLOCK_N, sa, sb, C + j * M + i, M);
            }
        }
    }

    free(buffer);
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
 * Column-major matrix
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
