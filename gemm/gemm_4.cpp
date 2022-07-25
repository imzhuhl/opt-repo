/**
 * block
 */
#include <arm_neon.h>
#include <cstdlib>

void packA_8(int M, int N, float *A, int lda, float *sa) {
    float *ptr_sa = sa;
    for (int i = 0; i < M; i += 8) {
        for (int j = 0; j < N; j++) {
            ptr_sa[0] = A[i * lda + j];
            ptr_sa[1] = A[(i + 1) * lda + j];
            ptr_sa[2] = A[(i + 2) * lda + j];
            ptr_sa[3] = A[(i + 3) * lda + j];
            ptr_sa[4] = A[(i + 4) * lda + j];
            ptr_sa[5] = A[(i + 5) * lda + j];
            ptr_sa[6] = A[(i + 6) * lda + j];
            ptr_sa[7] = A[(i + 7) * lda + j];
            ptr_sa += 8;
        }
    }
}

void packB_8(int M, int N, float *B, int ldb, float *sb) {
    float *ptr_sb = sb;

    for (int j = 0; j < N; j += 8) {
        for (int i = 0; i < M; i++) {
            ptr_sb[0] = B[i * ldb + j];
            ptr_sb[1] = B[i * ldb + j + 1];
            ptr_sb[2] = B[i * ldb + j + 2];
            ptr_sb[3] = B[i * ldb + j + 3];
            ptr_sb[4] = B[i * ldb + j + 4];
            ptr_sb[5] = B[i * ldb + j + 5];
            ptr_sb[6] = B[i * ldb + j + 6];
            ptr_sb[7] = B[i * ldb + j + 7];
            ptr_sb[8] = B[i * ldb + j + 8];
            ptr_sb += 8;
        }
    }
}

void fmla_8x8(int K, float *sa, float *sb, float *C, int ldc) {
    __asm__ __volatile__(
        "mov x4, %[sa];"
        "mov x5, %[sb];"
        "mov x6, %[k];"
        "lsl %[ldc], %[ldc], #2;"  // ldc = ldc * 4
        "mov x10, %[C];"
        "add x11, x10, %[ldc];"
        "add x12, x11, %[ldc];"
        "add x13, x12, %[ldc];"
        "add x14, x13, %[ldc];"
        "add x15, x14, %[ldc];"
        "add x16, x15, %[ldc];"
        "add x17, x16, %[ldc];"
        "ldr q10, [x10];"
        "ldr q11, [x10, #16];"
        "ldr q12, [x11];"
        "ldr q13, [x11, #16];"
        "ldr q14, [x12];"
        "ldr q15, [x12, #16];"
        "ldr q16, [x13];"
        "ldr q17, [x13, #16];"
        "ldr q18, [x14];"
        "ldr q19, [x14, #16];"
        "ldr q20, [x15];"
        "ldr q21, [x15, #16];"
        "ldr q22, [x16];"
        "ldr q23, [x16, #16];"
        "ldr q24, [x17];"
        "ldr q25, [x17, #16];"
        "1:"  // main loop
        "ldr q0, [x4];"
        "ldr q1, [x4, #16];"
        "ldr q3, [x5];"
        "ldr q4, [x5, #16];"
        "fmla v10.4s, v3.4s, v0.s[0];"
        "fmla v12.4s, v3.4s, v0.s[1];"
        "fmla v14.4s, v3.4s, v0.s[2];"
        "fmla v16.4s, v3.4s, v0.s[3];"
        "fmla v11.4s, v4.4s, v0.s[0];"
        "fmla v13.4s, v4.4s, v0.s[1];"
        "fmla v15.4s, v4.4s, v0.s[2];"
        "fmla v17.4s, v4.4s, v0.s[3];"
        "fmla v18.4s, v3.4s, v1.s[0];"
        "fmla v19.4s, v4.4s, v1.s[0];"
        "fmla v20.4s, v3.4s, v1.s[1];"
        "fmla v21.4s, v4.4s, v1.s[1];"
        "fmla v22.4s, v3.4s, v1.s[2];"
        "fmla v23.4s, v4.4s, v1.s[2];"
        "fmla v24.4s, v3.4s, v1.s[3];"
        "fmla v25.4s, v4.4s, v1.s[3];"
        "add x4, x4, #32;"
        "add x5, x5, #32;"
        "sub x6, x6, #1;"
        "cmp x6, #0;"
        "bgt 1b;"
        
        "str q10, [x10];"
        "str q11, [x10, #16];"
        "str q12, [x11];"
        "str q13, [x11, #16];"
        "str q14, [x12];"
        "str q15, [x12, #16];"
        "str q16, [x13];"
        "str q17, [x13, #16];"
        "str q18, [x14];"
        "str q19, [x14, #16];"
        "str q20, [x15];"
        "str q21, [x15, #16];"
        "str q22, [x16];"
        "str q23, [x16, #16];"
        "str q24, [x17];"
        "str q25, [x17, #16];"

        : [C] "+&r"(C), [ldc] "+&r"(ldc)
        : [sa] "r"(sa), [sb] "r"(sb), [k] "r"(K)
        : "cc", "memory", "p0", "x4", "x5", "x6", "x10", "x11", "x12", "x13", "x14", "x15", "x16",
          "x17", "v0", "v1", "v3", "v4", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
          "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25");
}

void kernel_8x8(int M, int K, int N, float *sa, float *sb, float *C, int ldc) {
    for (int i = 0; i < M; i += 8) {
        for (int j = 0; j < N; j += 8) {
            fmla_8x8(K, &sa[i * K], &sb[j * K], &C[i * ldc + j], ldc);
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
    float *buffer = (float *)malloc((BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * sizeof(float));

    float *sa = buffer;
    float *sb = buffer + BLOCK_M * BLOCK_K;

    for (int i = 0; i < M; i += BLOCK_M) {
        for (int p = 0; p < K; p += BLOCK_K) {
            packA_8(BLOCK_M, BLOCK_K, A + i * K + p, K, sa);
            for (int j = 0; j < N; j += BLOCK_N) {
                packB_8(BLOCK_K, BLOCK_N, B + p * N + j, N, sb);
                kernel_8x8(BLOCK_M, BLOCK_K, BLOCK_N, sa, sb, C + i * N + j, N);
            }
        }
    }

    free(buffer);
    return 0;
}
