#include <iostream>
#include <arm_neon.h>

void packA(int M, int N, float *A, int lda, float *sa) {
    float *ptr_sa = sa;
    float32x4_t v0;

    for (int i = 0; i < M; i += 4) {
        for (int j = 0; j < N; j++) {
            v0 = vld1q_f32(&A[j * lda + i]); 
            vst1q_f32(ptr_sa, v0);
            ptr_sa += 4;
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

//              00 01 02 03
//              .  .  .  .
// 
// 00 ...       00 01 02 03
// 10 ...       10 11 12 13
// 20 ...       20 21 22 23
// 30 ...       30 31 32 33
// 
void kernel(int M, int K, int N, float *A, float *B, float *C, int ldc) {
    float32x4_t	va0;
    float32x4_t vb0, vb1, vb2, vb3;
    float32x4_t vc0, vc1, vc2, vc3;
    float *ptr_a, *ptr_b, *ptr_c;
    float *ptr_c0, *ptr_c1, *ptr_c2, *ptr_c3;

    for (int j = 0; j < N; j += 4) { 
        for (int i = 0; i < M; i += 4) {
            ptr_c0 = &C[j * ldc + i]; 
            ptr_c1 = ptr_c0 + ldc;
            ptr_c2 = ptr_c1 + ldc;
            ptr_c3 = ptr_c2 + ldc;
            vc0 = vld1q_f32(ptr_c0);
            vc1 = vld1q_f32(ptr_c1);
            vc2 = vld1q_f32(ptr_c2);
            vc3 = vld1q_f32(ptr_c3);
            ptr_a = &A[i * K];
            ptr_b = &B[j * K];
            for (int p = 0; p < K; p++) {
                va0 = vld1q_f32(ptr_a);
                vb0 = vld1q_dup_f32(ptr_b);
                vb1 = vld1q_dup_f32(ptr_b + 1);
                vb2 = vld1q_dup_f32(ptr_b + 2);
                vb3 = vld1q_dup_f32(ptr_b + 3);

                vc0 = vfmaq_f32(vc0, va0, vb0);
                vc1 = vfmaq_f32(vc1, va0, vb1);
                vc2 = vfmaq_f32(vc2, va0, vb2);
                vc3 = vfmaq_f32(vc3, va0, vb3);

                ptr_a += 4;
                ptr_b += 4;
            }
            vst1q_f32(ptr_c0, vc0);
            vst1q_f32(ptr_c1, vc1);
            vst1q_f32(ptr_c2, vc2);
            vst1q_f32(ptr_c3, vc3);
        }
    }
}

#ifdef DEBUG
constexpr int BLOCK_M = 8;
constexpr int BLOCK_K = 8;
constexpr int BLOCK_N = 8;
#else
constexpr int BLOCK_M = 32;
constexpr int BLOCK_K = 32;
constexpr int BLOCK_N = 32;
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