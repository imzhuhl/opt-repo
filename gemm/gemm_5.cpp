#include <arm_neon.h>
#include <iostream>

void packA(int M, int N, float *A, int lda, float *sa) {
    float *ptr_sa = sa;
    float32x4_t v0, v1, v2, v3;

    for (int i = 0; i < M; i += 16) {
        for (int j = 0; j < N; j++) {
            v0 = vld1q_f32(&A[j * lda + i]);
            v1 = vld1q_f32(&A[j * lda + i + 4]);
            v2 = vld1q_f32(&A[j * lda + i + 8]);
            v3 = vld1q_f32(&A[j * lda + i + 12]);
            vst1q_f32(ptr_sa, v0);
            vst1q_f32(ptr_sa + 4, v1);
            vst1q_f32(ptr_sa + 8, v2);
            vst1q_f32(ptr_sa + 12, v3);
            ptr_sa += 16;
        }
    }
}

void packB(int K, int N, float *B, int ldb, float *sb) {
    float *ptr_sb = sb;
    for (int j = 0; j < N; j += 4) {
        for (int i = 0; i < K; i++) {
            *(ptr_sb + 0) = B[(j + 0) * ldb + i];
            *(ptr_sb + 1) = B[(j + 1) * ldb + i];
            *(ptr_sb + 2) = B[(j + 2) * ldb + i];
            *(ptr_sb + 3) = B[(j + 3) * ldb + i];
            ptr_sb += 4;
        }
    }
}

#define LOAD_C_16x4                \
    vc00 = vld1q_f32(ptr_c0);      \
    vc01 = vld1q_f32(ptr_c1);      \
    vc02 = vld1q_f32(ptr_c2);      \
    vc03 = vld1q_f32(ptr_c3);      \
    vc10 = vld1q_f32(ptr_c0 + 4);  \
    vc11 = vld1q_f32(ptr_c1 + 4);  \
    vc12 = vld1q_f32(ptr_c2 + 4);  \
    vc13 = vld1q_f32(ptr_c3 + 4);  \
    vc20 = vld1q_f32(ptr_c0 + 8);  \
    vc21 = vld1q_f32(ptr_c1 + 8);  \
    vc22 = vld1q_f32(ptr_c2 + 8);  \
    vc23 = vld1q_f32(ptr_c3 + 8);  \
    vc30 = vld1q_f32(ptr_c0 + 12); \
    vc31 = vld1q_f32(ptr_c1 + 12); \
    vc32 = vld1q_f32(ptr_c2 + 12); \
    vc33 = vld1q_f32(ptr_c3 + 12);

#define LOAD_A_4                \
    va0 = vld1q_f32(ptr_a);     \
    va1 = vld1q_f32(ptr_a + 4); \
    va2 = vld1q_f32(ptr_a + 8); \
    va3 = vld1q_f32(ptr_a + 12);

#define LOAD_B_4                    \
    vb0 = vld1q_dup_f32(ptr_b);     \
    vb1 = vld1q_dup_f32(ptr_b + 1); \
    vb2 = vld1q_dup_f32(ptr_b + 2); \
    vb3 = vld1q_dup_f32(ptr_b + 3);

#define MATMUL_16x4                   \
    vc00 = vfmaq_f32(vc00, va0, vb0); \
    vc01 = vfmaq_f32(vc01, va0, vb1); \
    vc02 = vfmaq_f32(vc02, va0, vb2); \
    vc03 = vfmaq_f32(vc03, va0, vb3); \
    vc10 = vfmaq_f32(vc10, va1, vb0); \
    vc11 = vfmaq_f32(vc11, va1, vb1); \
    vc12 = vfmaq_f32(vc12, va1, vb2); \
    vc13 = vfmaq_f32(vc13, va1, vb3); \
    vc20 = vfmaq_f32(vc20, va2, vb0); \
    vc21 = vfmaq_f32(vc21, va2, vb1); \
    vc22 = vfmaq_f32(vc22, va2, vb2); \
    vc23 = vfmaq_f32(vc23, va2, vb3); \
    vc30 = vfmaq_f32(vc30, va3, vb0); \
    vc31 = vfmaq_f32(vc31, va3, vb1); \
    vc32 = vfmaq_f32(vc32, va3, vb2); \
    vc33 = vfmaq_f32(vc33, va3, vb3);

#define STORE_C_16x4              \
    vst1q_f32(ptr_c0, vc00);      \
    vst1q_f32(ptr_c1, vc01);      \
    vst1q_f32(ptr_c2, vc02);      \
    vst1q_f32(ptr_c3, vc03);      \
    vst1q_f32(ptr_c0 + 4, vc10);  \
    vst1q_f32(ptr_c1 + 4, vc11);  \
    vst1q_f32(ptr_c2 + 4, vc12);  \
    vst1q_f32(ptr_c3 + 4, vc13);  \
    vst1q_f32(ptr_c0 + 8, vc20);  \
    vst1q_f32(ptr_c1 + 8, vc21);  \
    vst1q_f32(ptr_c2 + 8, vc22);  \
    vst1q_f32(ptr_c3 + 8, vc23);  \
    vst1q_f32(ptr_c0 + 12, vc30); \
    vst1q_f32(ptr_c1 + 12, vc31); \
    vst1q_f32(ptr_c2 + 12, vc32); \
    vst1q_f32(ptr_c3 + 12, vc33);

//
//              00 01 02 03
//              .  .  .  .
//
// 00 ...       00 01 02 03
// 10 ...       10 11 12 13
// 20 ...       20 21 22 23
// 30 ...       30 31 32 33
// 40 ...
// .
//
void kernel(int M, int K, int N, float *A, float *B, float *C, int ldc) {
    float32x4_t va0, va1, va2, va3;
    float32x4_t vb0, vb1, vb2, vb3;
    float32x4_t vc00, vc01, vc02, vc03, vc10, vc11, vc12, vc13, vc20, vc21, vc22, vc23, vc30, vc31, vc32,
        vc33;
    float *ptr_a, *ptr_b, *ptr_c;
    float *ptr_c0, *ptr_c1, *ptr_c2, *ptr_c3;

    for (int j = 0; j < N; j += 4) {
        for (int i = 0; i < M; i += 16) {
            ptr_c0 = &C[j * ldc + i];
            ptr_c1 = ptr_c0 + ldc;
            ptr_c2 = ptr_c1 + ldc;
            ptr_c3 = ptr_c2 + ldc;
            ptr_a = &A[i * K];
            ptr_b = &B[j * K];

            LOAD_C_16x4;

            for (int p = 0; p < K; p++) {
                LOAD_A_4;
                LOAD_B_4;
                MATMUL_16x4;
                ptr_a += 16;
                ptr_b += 4;
            }

            STORE_C_16x4;
        }
    }
}

#ifdef DEBUG
constexpr int BLOCK_M = 16;
constexpr int BLOCK_K = 16;
constexpr int BLOCK_N = 16;
#else
constexpr int BLOCK_M = 256;
constexpr int BLOCK_K = 256;
constexpr int BLOCK_N = 256;
#endif

int my_impl(int M, int K, int N, float *A, int lda, float *B, int ldb, float *C, int ldc) {
    float *sa = new float[BLOCK_M * BLOCK_K];
    float *sb = new float[BLOCK_K * BLOCK_N];

    for (int bm = 0; bm < M; bm += BLOCK_M) {
        for (int bk = 0; bk < K; bk += BLOCK_K) {
            packA(BLOCK_M, BLOCK_K, A + bk * M + bm, lda, sa);
            for (int bn = 0; bn < N; bn += BLOCK_N) {
                packB(BLOCK_K, BLOCK_N, B + bn * K + bk, ldb, sb);
                kernel(BLOCK_M, BLOCK_K, BLOCK_N, sa, sb, C + bn * M + bm, M);
            }
        }
    }
    return 0;
}
