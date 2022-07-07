/**
 * inline assembly code based on gemm_4.cpp
 * */

#include <iostream>
#include <arm_neon.h>

void packA_4(int M, int N, float *A, int lda, float *sa) {
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

void packB_4(int K, int N, float *B, int ldb, float *sb) {
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
void kernel_4x4(int K, float *sa, float *sb, float *C, int ldc) {
    __asm__ __volatile__ (
        // c ptr        
        "mov x2, %x[ldc];"
        "mov x3, %x[k];"
        "mov x10, %x[C];"
        "add x11, x10, x2;"
        "add x12, x11, x2;"
        "add x13, x12, x2;"
        "ld1 {v16.4s}, [x10];"
        "ld1 {v17.4s}, [x11];"
        "ld1 {v18.4s}, [x12];"
        "ld1 {v19.4s}, [x13];"
        
        "1:"

        "ld1 {v8.4s}, [%x[sb]];"
        "add %x[sb], %x[sb], #16;"
        "ld1 {v0.4s}, [%x[sa]];"
        "add %x[sa], %x[sa], #16;"

        "fmla v16.4s, v0.4s, v8.s[0]; "
        "fmla v17.4s, v0.4s, v8.s[1]; "
        "fmla v18.4s, v0.4s, v8.s[2]; "
        "fmla v19.4s, v0.4s, v8.s[3]; "

        "subs x3, x3, #1;"
        "bgt 1b;"

        // store
        "st1 {v16.4s}, [x10];" 
        "st1 {v17.4s}, [x11];" 
        "st1 {v18.4s}, [x12];" 
        "st1 {v19.4s}, [x13];" 

        : [C] "+&r" (C), [sa] "+&r" (sa), [sb] "+&r" (sb)
        : [k] "r" (K), [ldc] "r" (ldc * 4)
        : "cc", "memory", "x2", "x3", "x10", "x11", "x12", "x13", "v0", "v8", "v16", "v17", "v18", "v19"
    );
}

void kernel(int M, int K, int N, float *A, float *B, float *C, int ldc) {
    float32x4_t	va0;
    float32x4_t vb0, vb1, vb2, vb3;
    float32x4_t vc0, vc1, vc2, vc3;
    float *ptr_a, *ptr_b, *ptr_c;
    float *ptr_c0, *ptr_c1, *ptr_c2, *ptr_c3;

    for (int j = 0; j < N; j += 4) { 
        for (int i = 0; i < M; i += 4) {
            kernel_4x4(K, A + i * K, B + j * K, C + j * ldc + i, ldc);
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

int my_impl(int M, int K, int N, float *A, int lda, float *B, int ldb, float *C, int ldc) {
    
    float *sa = new float[BLOCK_M * BLOCK_K]; 
    float *sb = new float[BLOCK_K * BLOCK_N];

    for (int bm = 0; bm < M; bm += BLOCK_M) {
        for (int bk = 0; bk < K; bk += BLOCK_K) {
            packA_4(BLOCK_M, BLOCK_K, A + bk * M + bm, lda, sa);
            for (int bn = 0; bn < N; bn += BLOCK_N) {
                packB_4(BLOCK_K, BLOCK_N, B + bn * K + bk, ldb, sb);
                kernel(BLOCK_M, BLOCK_K, BLOCK_N, sa, sb, C + bn * M + bm, M);
            }
        }
    }

    delete[] sa;
    delete[] sb;
    return 0;
}