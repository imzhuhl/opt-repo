/**
 * based on gemm_1
 * use more registers
 */

#include <arm_neon.h>
#include <cstdlib>
#include <iostream>

void packA(int M, int N, float *A, float *sa) {
    float *ptr_sa = sa;
    for (int i = 0; i < M; i += 4) {
        for (int j = 0; j < N; j++) {
            ptr_sa[0] = A[i * N + j];
            ptr_sa[1] = A[(i + 1) * N + j];
            ptr_sa[2] = A[(i + 2) * N + j];
            ptr_sa[3] = A[(i + 3) * N + j];
            ptr_sa += 4;
        }
    }
}

void packB(int M, int N, float *B, float *sb) {
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

int my_impl(int M, int K, int N, float *A, float *B, float *C) {
    float *sa = (float *)malloc(M * K * sizeof(float));
    float *sb = (float *)malloc(K * N * sizeof(float));

    float32x4_t va, vb0, vb1;
    float32x4_t vc00, vc01, vc10, vc11, vc20, vc21, vc30, vc31;

    packA(M, K, A, sa);
    packB(K, N, B, sb);

    for (int i = 0; i < M; i += 4) {
        for (int j = 0; j < N; j += 8) {

            vc00 = vld1q_f32(&C[i * N + j]);
            vc01 = vld1q_f32(&C[i * N + j + 4]);
            vc10 = vld1q_f32(&C[(i + 1) * N + j]);
            vc11 = vld1q_f32(&C[(i + 1) * N + j + 4]);
            vc20 = vld1q_f32(&C[(i + 2) * N + j]);
            vc21 = vld1q_f32(&C[(i + 2) * N + j + 4]);
            vc30 = vld1q_f32(&C[(i + 3) * N + j]);
            vc31 = vld1q_f32(&C[(i + 3) * N + j + 4]);

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

            vst1q_f32(&C[i * N + j], vc00);
            vst1q_f32(&C[i * N + j + 4], vc01);
            vst1q_f32(&C[(i + 1) * N + j], vc10);
            vst1q_f32(&C[(i + 1) * N + j + 4], vc11);
            vst1q_f32(&C[(i + 2) * N + j], vc20);
            vst1q_f32(&C[(i + 2) * N + j + 4], vc21);
            vst1q_f32(&C[(i + 3) * N + j], vc30);
            vst1q_f32(&C[(i + 3) * N + j + 4], vc31);
        }
    }
    free(sa);
    free(sb);
    return 0;
}
