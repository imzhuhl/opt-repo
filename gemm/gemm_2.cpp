/**
 * block
 */ 
#include <arm_neon.h>
#include <cstdlib>

void packA(int M, int N, float *A, int lda, float *sa) {
    float *ptr_sa = sa;
    for (int i = 0; i < M; i += 4) {
        for (int j = 0; j < N; j++) {
            ptr_sa[0] = A[i * lda + j];
            ptr_sa[1] = A[(i + 1) * lda + j];
            ptr_sa[2] = A[(i + 2) * lda + j];
            ptr_sa[3] = A[(i + 3) * lda + j];
            ptr_sa += 4;
        }
    }
}

void packB(int M, int N, float *B, int ldb, float *sb) {
    float *ptr_sb = sb;
    float32x4_t vb0, vb1;

    for (int j = 0; j < N; j += 8) {
        for (int i = 0; i < M; i++) {
            vb0 = vld1q_f32(&B[i * N + j]);
            vb1 = vld1q_f32(&B[i * N + j + 4]);
            vst1q_f32(ptr_sb, vb0); 
            vst1q_f32(ptr_sb + 4, vb1);
            ptr_sb += 8;
        }
    }
}

void kernel_4x8(int M, int K, int N, float *sa, float *sb, float *C, int ldc) {
    float32x4_t va, vb0, vb1;
    float32x4_t vc00, vc01, vc10, vc11, vc20, vc21, vc30, vc31;

    for (int i = 0; i < M; i += 4) {
        for (int j = 0; j < N; j += 8) {
            vc00 = vld1q_f32(&C[i * ldc + j]);
            vc01 = vld1q_f32(&C[i * ldc + j + 4]);
            vc10 = vld1q_f32(&C[(i + 1) * ldc + j]);
            vc11 = vld1q_f32(&C[(i + 1) * ldc + j + 4]);
            vc20 = vld1q_f32(&C[(i + 2) * ldc + j]);
            vc21 = vld1q_f32(&C[(i + 2) * ldc + j + 4]);
            vc30 = vld1q_f32(&C[(i + 3) * ldc + j]);
            vc31 = vld1q_f32(&C[(i + 3) * ldc + j + 4]);

            for (int p = 0; p < K; p++) {
                va = vld1q_f32(&sa[i * K + 4 * p]);
                vb0 = vld1q_f32(&sb[j * K + 8 * p]);
                vb1 = vld1q_f32(&sb[j * K + 8 * p + 4]);

                vc00 = vfmaq_n_f32(vc00, vb0, va[0]);
                vc01 = vfmaq_n_f32(vc01, vb1, va[0]);
                vc10 = vfmaq_n_f32(vc10, vb0, va[1]);
                vc11 = vfmaq_n_f32(vc11, vb1, va[1]);
                vc20 = vfmaq_n_f32(vc20, vb0, va[2]);
                vc21 = vfmaq_n_f32(vc21, vb1, va[2]);
                vc30 = vfmaq_n_f32(vc30, vb0, va[3]);
                vc31 = vfmaq_n_f32(vc31, vb1, va[3]);
            }

            vst1q_f32(&C[i * ldc + j], vc00);
            vst1q_f32(&C[i * ldc + j + 4], vc01);
            vst1q_f32(&C[(i + 1) * ldc + j], vc10);
            vst1q_f32(&C[(i + 1) * ldc + j + 4], vc11);
            vst1q_f32(&C[(i + 2) * ldc + j], vc20);
            vst1q_f32(&C[(i + 2) * ldc + j + 4], vc21);
            vst1q_f32(&C[(i + 3) * ldc + j], vc30);
            vst1q_f32(&C[(i + 3) * ldc + j + 4], vc31);
        }
    }
}

#ifdef DEBUG
constexpr int BLOCK_M = 8;
constexpr int BLOCK_K = 8;
constexpr int BLOCK_N = 8;
#else
constexpr int BLOCK_M = 256;
constexpr int BLOCK_K = 256;
constexpr int BLOCK_N = 256;
#endif

int my_impl(int M, int K, int N, float *A, float *B, float *C) {
    float *buffer = (float *) malloc ((BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * sizeof(float));

    float *sa = buffer;
    float *sb = buffer + BLOCK_M * BLOCK_K;

    for (int i = 0; i < M; i += BLOCK_M) {
        for (int p = 0; p < K; p += BLOCK_K) {
            packA(BLOCK_M, BLOCK_K, A + i * K + p, K, sa);
            for (int j = 0; j < N; j += BLOCK_N) {
                packB(BLOCK_K, BLOCK_N, B + p * N + j, N, sb);
                kernel_4x8(BLOCK_M, BLOCK_K, BLOCK_N, sa, sb, C + i * N + j, N);
            }
        }
    }

    free(buffer);
    return 0;
}
