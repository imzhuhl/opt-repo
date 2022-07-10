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

    for (int j = 0; j < N; j += 4) {
        for (int i = 0; i < M; i++) {
            ptr_sb[0] = B[i * ldb + j];
            ptr_sb[1] = B[i * ldb + j + 1];
            ptr_sb[2] = B[i * ldb + j + 2];
            ptr_sb[3] = B[i * ldb + j + 3];
            ptr_sb += 4;
        }
    }
}

void kernel_4x4(int M, int K, int N, float *sa, float *sb, float *C, int ldc) {
    float32x4_t va, vb;
    float32x4_t vc0, vc1, vc2, vc3;

    for (int i = 0; i < M; i += 4) {
        for (int j = 0; j < N; j += 4) {
            vc0 = vld1q_f32(&C[i * ldc + j]);
            vc1 = vld1q_f32(&C[(i + 1) * ldc + j]);
            vc2 = vld1q_f32(&C[(i + 2) * ldc + j]);
            vc3 = vld1q_f32(&C[(i + 3) * ldc + j]);
            for (int p = 0; p < K; p++) {
                va = vld1q_f32(&sa[i * K + 4 * p]);
                vb = vld1q_f32(&sb[j * K + 4 * p]);

                vc0 = vfmaq_n_f32(vc0, vb, va[0]);
                vc1 = vfmaq_n_f32(vc1, vb, va[1]);
                vc2 = vfmaq_n_f32(vc2, vb, va[2]);
                vc3 = vfmaq_n_f32(vc3, vb, va[3]);
            }

            vst1q_f32(&C[i * ldc + j], vc0);
            vst1q_f32(&C[(i + 1) * ldc + j], vc1);
            vst1q_f32(&C[(i + 2) * ldc + j], vc2);
            vst1q_f32(&C[(i + 3) * ldc + j], vc3);
        }
    }
}

#ifdef DEBUG
constexpr int BLOCK_M = 4;
constexpr int BLOCK_K = 4;
constexpr int BLOCK_N = 4;
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
                kernel_4x4(BLOCK_M, BLOCK_K, BLOCK_N, sa, sb, C + i * N + j, N);
            }
        }
    }

    free(buffer);
    return 0;
}
