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

    for (int j = 0; j < N; j += 4) {
        for (int i = 0; i < M; i++) {
            ptr_sb[0] = B[i * N + j];
            ptr_sb[1] = B[i * N + j + 1];
            ptr_sb[2] = B[i * N + j + 2];
            ptr_sb[3] = B[i * N + j + 3];
            ptr_sb += 4;
        }
    }
}

int my_impl(int M, int K, int N, float *A, float *B, float *C) {
    float *sa = (float *)malloc(M * K * sizeof(float));
    float *sb = (float *)malloc(K * N * sizeof(float));

    float32x4_t va, vb;
    float32x4_t vc0, vc1, vc2, vc3;

    packA(M, K, A, sa);
    packB(K, N, B, sb);

    for (int i = 0; i < M; i += 4) {
        for (int j = 0; j < N; j += 4) {
            vc0 = vld1q_f32(&C[i * N + j]);
            vc1 = vld1q_f32(&C[(i + 1) * N + j]);
            vc2 = vld1q_f32(&C[(i + 2) * N + j]);
            vc3 = vld1q_f32(&C[(i + 3) * N + j]);
            for (int p = 0; p < K; p++) {
                va = vld1q_f32(&sa[i * K + 4 * p]);
                vb = vld1q_f32(&sb[j * K + 4 * p]);

                vc0 = vfmaq_n_f32(vc0, vb, va[0]);
                vc1 = vfmaq_n_f32(vc1, vb, va[1]);
                vc2 = vfmaq_n_f32(vc2, vb, va[2]);
                vc3 = vfmaq_n_f32(vc3, vb, va[3]);
            }

            vst1q_f32(&C[i * N + j], vc0);
            vst1q_f32(&C[(i + 1) * N + j], vc1);
            vst1q_f32(&C[(i + 2) * N + j], vc2);
            vst1q_f32(&C[(i + 3) * N + j], vc3);
        }
    }
    free(sa);
    free(sb);
    return 0;
}
