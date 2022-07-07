/**
 * inline assembly code based on gemm_4.cpp
 * */

#include <iostream>
#include <arm_neon.h>

void packA_16(int M, int N, float *A, int lda, float *sa) {
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

void kernel_16x4(int K, float *sa, float *sb, float *C, int ldc) {
    __asm__ __volatile__ (
        // x5: ldc, x6: k
        "mov x5, %x[ldc];"
        "mov x6, %x[k];"

        // x10, x11, x12, x13: column 0, 1, 2, 3 of C
        "mov x10, %x[C];"
        "add x11, x10, x5;"
        "add x12, x11, x5;"
        "add x13, x12, x5;"

        "ld1 { v0.4s,  v1.4s,  v2.4s,  v3.4s },  [x10];"
        "ld1 { v4.4s,  v5.4s,  v6.4s,  v7.4s },  [x11];"
        "ld1 { v8.4s,  v9.4s,  v10.4s, v11.4s }, [x12];"
        "ld1 { v12.4s, v13.4s, v14.4s, v15.4s }, [x13];"
        
        "1:"

        "ld1 {v20.4s}, [%x[sb]];"
        "add %x[sb], %x[sb], #16;"
        "ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [%x[sa]];"
        "add %x[sa], %x[sa], #64;"

        "fmla  v0.4s,  v16.4s, v20.s[0]; "
        "fmla  v4.4s,  v16.4s, v20.s[1]; "
        "fmla  v8.4s,  v16.4s, v20.s[2]; "
        "fmla  v12.4s, v16.4s, v20.s[3]; "
        
        "fmla v1.4s,  v17.4s, v20.s[0]; "
        "fmla v5.4s,  v17.4s, v20.s[1]; "
        "fmla v9.4s,  v17.4s, v20.s[2]; "
        "fmla v13.4s, v17.4s, v20.s[3]; "

        "fmla v2.4s,  v18.4s, v20.s[0]; "
        "fmla v6.4s,  v18.4s, v20.s[1]; "
        "fmla v10.4s, v18.4s, v20.s[2]; "
        "fmla v14.4s, v18.4s, v20.s[3]; "

        "fmla v3.4s,  v19.4s, v20.s[0]; "
        "fmla v7.4s,  v19.4s, v20.s[1]; "
        "fmla v11.4s, v19.4s, v20.s[2]; "
        "fmla v15.4s, v19.4s, v20.s[3]; "

        "subs x6, x6, #1;"
        "bgt 1b;"

        // store
        "st1 { v0.4s,  v1.4s,  v2.4s,  v3.4s },  [x10];" 
        "st1 { v4.4s,  v5.4s,  v6.4s,  v7.4s },  [x11];" 
        "st1 { v8.4s,  v9.4s,  v10.4s, v11.4s }, [x12];" 
        "st1 { v12.4s, v13.4s, v14.4s, v15.4s }, [x13];" 

        : [C] "+&r" (C), [sa] "+&r" (sa), [sb] "+&r" (sb)
        : [k] "r" (K), [ldc] "r" (ldc * 4)
        : "cc", "memory", "x5", "x6", "x10", "x11", "x12", "x13", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20"
    );
}

void kernel(int M, int K, int N, float *A, float *B, float *C, int ldc) {
    float32x4_t	va0;
    float32x4_t vb0, vb1, vb2, vb3;
    float32x4_t vc0, vc1, vc2, vc3;
    float *ptr_a, *ptr_b, *ptr_c;
    float *ptr_c0, *ptr_c1, *ptr_c2, *ptr_c3;

    for (int j = 0; j < N; j += 4) { 
        for (int i = 0; i < M; i += 16) {
            kernel_16x4(K, A + i * K, B + j * K, C + j * ldc + i, ldc);
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
    
    float *sa = new (std::align_val_t(4)) float[BLOCK_M * BLOCK_K];
    float *sb = new (std::align_val_t(4)) float[BLOCK_K * BLOCK_N];

    for (int bm = 0; bm < M; bm += BLOCK_M) {
        for (int bk = 0; bk < K; bk += BLOCK_K) {
            packA_16(BLOCK_M, BLOCK_K, A + bk * M + bm, lda, sa);
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