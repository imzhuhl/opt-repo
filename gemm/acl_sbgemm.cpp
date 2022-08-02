#include <arm_neon.h>
#include <chrono>
#include <iostream>
#include "utils.hpp"

constexpr int MIN_SIZE = 2048;
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

void packA(int M, int N, bfloat16 *A, int lda, bfloat16 *sa) {
    uint16x4_t v0;
    for (int i = 0; i < M; i += 8) {
        for (int j = 0; j < N; j += 4) {
            for (int ii = 0; ii < 8; ii++) {
                v0 = vld1_u16(reinterpret_cast<uint16_t *>(&A[(i + ii) * lda + j]));
                vst1_u16(reinterpret_cast<uint16_t *>(sa + ii * 4), v0);
            }

            sa += 32;
        }
    }
}

void packB(int M, int N, bfloat16 *B, int ldb, bfloat16 *sb) {
    for (int j = 0; j < N; j += 12) {
        for (int i = 0; i < M; i += 4) {
            for (int jj = 0; jj < 12; jj++) {
                for (int ii = 0; ii < 4; ii++) {
                    sb[jj * 4 + ii] = B[(i + ii) * ldb + j + jj];
                }
            }
            sb += 48;
        }
    }

#if 0
    sb = sb - M * N;
    for (int i = 0; i < M * N; i++) {
        printf(" %lf\n", bf16_to_fp32(sb[i]));
    }
#endif
}

void a64_interleaved_bf16fp32_mmla_8x12(
    const bfloat16 *Apanel, const bfloat16 *Bpanel,
    float *Cpanel, int ablocks, int bblocks, int K) {

    struct KernelArgs {
        size_t bblocks = {};
        size_t K = {};
        const bfloat16 *Bpanel = {};
    } ka;

    ka.bblocks = bblocks;
    ka.K = (K/4) - 1;
    ka.Bpanel = Bpanel;

    __asm__ __volatile__(

      "1:"  // Height loop
      "ldr x22, [%x[args_ptr], %[offsetof_bblocks]]\n"
      "mov x21, %x[Apanel]\n"
      "ldr x20, [%x[args_ptr], %[offsetof_Bpanel]]\n"
      "2:"  // Width loop
      "ldr x19, [%x[args_ptr], %[offsetof_K]]\n"
      "mov %x[Apanel], x21\n"
      "cmp x19, #0x2\n"
      "movi v8.16b, #0x0\n"
      "movi v9.16b, #0x0\n"
      "ldr q4, [x20, #0x0]\n"
      "movi v10.16b, #0x0\n"
      "movi v11.16b, #0x0\n"
      "ldr q0, [%x[Apanel], #0x0]\n"
      "movi v12.16b, #0x0\n"
      "movi v13.16b, #0x0\n"
      "ldr q1, [%x[Apanel], #0x10]\n"
      "movi v14.16b, #0x0\n"
      "movi v15.16b, #0x0\n"
      "ldr q5, [x20, #0x10]\n"
      "movi v16.16b, #0x0\n"
      "movi v17.16b, #0x0\n"
      "ldr q2, [%x[Apanel], #0x20]\n"
      "movi v18.16b, #0x0\n"
      "movi v19.16b, #0x0\n"
      "add x20, x20, #0x20\n"
      "movi v20.16b, #0x0\n"
      "movi v21.16b, #0x0\n"
      "add %x[Apanel], %x[Apanel], #0x30\n"
      "movi v22.16b, #0x0\n"
      "movi v23.16b, #0x0\n"
      "movi v24.16b, #0x0\n"
      "movi v25.16b, #0x0\n"
      "movi v26.16b, #0x0\n"
      "movi v27.16b, #0x0\n"
      "movi v28.16b, #0x0\n"
      "movi v29.16b, #0x0\n"
      "movi v30.16b, #0x0\n"
      "movi v31.16b, #0x0\n"
      "blt 4f\n"
      "3:"  // main loop head
      "ldr q3, [%x[Apanel], #0x0]\n"
      ".inst 0x6e44ec08  // bfmmla v8.4s, v0.8h, v4.8h\n"
      ".inst 0x6e44ec2e  // bfmmla v14.4s, v1.8h, v4.8h\n"
      ".inst 0x6e45ec0b  // bfmmla v11.4s, v0.8h, v5.8h\n"
      ".inst 0x6e45ec31  // bfmmla v17.4s, v1.8h, v5.8h\n"
      "ldr q6, [x20, #0x0]\n"
      ".inst 0x6e44ec54  // bfmmla v20.4s, v2.8h, v4.8h\n"
      ".inst 0x6e45ec57  // bfmmla v23.4s, v2.8h, v5.8h\n"
      "ldr q7, [x20, #0x10]\n"
      ".inst 0x6e44ec7a  // bfmmla v26.4s, v3.8h, v4.8h\n"
      ".inst 0x6e45ec7d  // bfmmla v29.4s, v3.8h, v5.8h\n"
      "ldr q4, [x20, #0x20]\n"
      "ldr q5, [x20, #0x30]\n"
      ".inst 0x6e46ec09  // bfmmla v9.4s, v0.8h, v6.8h\n"
      ".inst 0x6e46ec2f  // bfmmla v15.4s, v1.8h, v6.8h\n"
      ".inst 0x6e46ec55  // bfmmla v21.4s, v2.8h, v6.8h\n"
      ".inst 0x6e46ec7b  // bfmmla v27.4s, v3.8h, v6.8h\n"
      "ldr q6, [x20, #0x40]\n"
      ".inst 0x6e47ec0c  // bfmmla v12.4s, v0.8h, v7.8h\n"
      ".inst 0x6e44ec0a  // bfmmla v10.4s, v0.8h, v4.8h\n"
      "sub x19, x19, #0x2\n"
      ".inst 0x6e45ec0d  // bfmmla v13.4s, v0.8h, v5.8h\n"
      ".inst 0x6e47ec32  // bfmmla v18.4s, v1.8h, v7.8h\n"
      "ldr q0, [%x[Apanel], #0x10]\n"
      ".inst 0x6e44ec30  // bfmmla v16.4s, v1.8h, v4.8h\n"
      ".inst 0x6e45ec33  // bfmmla v19.4s, v1.8h, v5.8h\n"
      "ldr q1, [%x[Apanel], #0x20]\n"
      ".inst 0x6e47ec58  // bfmmla v24.4s, v2.8h, v7.8h\n"
      ".inst 0x6e47ec7e  // bfmmla v30.4s, v3.8h, v7.8h\n"
      "ldr q7, [x20, #0x50]\n"
      ".inst 0x6e44ec56  // bfmmla v22.4s, v2.8h, v4.8h\n"
      ".inst 0x6e45ec59  // bfmmla v25.4s, v2.8h, v5.8h\n"
      "ldr q2, [%x[Apanel], #0x30]\n"
      ".inst 0x6e44ec7c  // bfmmla v28.4s, v3.8h, v4.8h\n"
      ".inst 0x6e45ec7f  // bfmmla v31.4s, v3.8h, v5.8h\n"
      "ldr q3, [%x[Apanel], #0x40]\n"
      ".inst 0x6e46ec08  // bfmmla v8.4s, v0.8h, v6.8h\n"
      ".inst 0x6e46ec2e  // bfmmla v14.4s, v1.8h, v6.8h\n"
      "ldr q4, [x20, #0x60]\n"
      ".inst 0x6e47ec0b  // bfmmla v11.4s, v0.8h, v7.8h\n"
      ".inst 0x6e47ec31  // bfmmla v17.4s, v1.8h, v7.8h\n"
      "ldr q5, [x20, #0x70]\n"
      ".inst 0x6e46ec54  // bfmmla v20.4s, v2.8h, v6.8h\n"
      ".inst 0x6e47ec57  // bfmmla v23.4s, v2.8h, v7.8h\n"
      "cmp x19, #0x2\n"
      ".inst 0x6e46ec7a  // bfmmla v26.4s, v3.8h, v6.8h\n"
      ".inst 0x6e47ec7d  // bfmmla v29.4s, v3.8h, v7.8h\n"
      "ldr q6, [x20, #0x80]\n"
      "ldr q7, [x20, #0x90]\n"
      ".inst 0x6e44ec09  // bfmmla v9.4s, v0.8h, v4.8h\n"
      ".inst 0x6e44ec2f  // bfmmla v15.4s, v1.8h, v4.8h\n"
      ".inst 0x6e44ec55  // bfmmla v21.4s, v2.8h, v4.8h\n"
      ".inst 0x6e44ec7b  // bfmmla v27.4s, v3.8h, v4.8h\n"
      "ldr q4, [x20, #0xa0]\n"
      ".inst 0x6e45ec0c  // bfmmla v12.4s, v0.8h, v5.8h\n"
      ".inst 0x6e46ec0a  // bfmmla v10.4s, v0.8h, v6.8h\n"
      ".inst 0x6e47ec0d  // bfmmla v13.4s, v0.8h, v7.8h\n"
      ".inst 0x6e45ec32  // bfmmla v18.4s, v1.8h, v5.8h\n"
      "ldr q0, [%x[Apanel], #0x50]\n"
      ".inst 0x6e46ec30  // bfmmla v16.4s, v1.8h, v6.8h\n"
      ".inst 0x6e47ec33  // bfmmla v19.4s, v1.8h, v7.8h\n"
      "ldr q1, [%x[Apanel], #0x60]\n"
      ".inst 0x6e45ec58  // bfmmla v24.4s, v2.8h, v5.8h\n"
      ".inst 0x6e45ec7e  // bfmmla v30.4s, v3.8h, v5.8h\n"
      "ldr q5, [x20, #0xb0]\n"
      ".inst 0x6e46ec56  // bfmmla v22.4s, v2.8h, v6.8h\n"
      ".inst 0x6e47ec59  // bfmmla v25.4s, v2.8h, v7.8h\n"
      "ldr q2, [%x[Apanel], #0x70]\n"
      ".inst 0x6e46ec7c  // bfmmla v28.4s, v3.8h, v6.8h\n"
      ".inst 0x6e47ec7f  // bfmmla v31.4s, v3.8h, v7.8h\n"
      "add %x[Apanel], %x[Apanel], #0x80\n"
      "add x20, x20, #0xc0\n"
      "bge 3b\n"
      "4:"  // main loop skip
      "ldr q3, [%x[Apanel], #0x0]\n"
      ".inst 0x6e44ec08  // bfmmla v8.4s, v0.8h, v4.8h\n"
      ".inst 0x6e44ec2e  // bfmmla v14.4s, v1.8h, v4.8h\n"
      ".inst 0x6e45ec0b  // bfmmla v11.4s, v0.8h, v5.8h\n"
      ".inst 0x6e45ec31  // bfmmla v17.4s, v1.8h, v5.8h\n"
      "ldr q6, [x20, #0x0]\n"
      ".inst 0x6e44ec54  // bfmmla v20.4s, v2.8h, v4.8h\n"
      ".inst 0x6e45ec57  // bfmmla v23.4s, v2.8h, v5.8h\n"
      "ldr q7, [x20, #0x10]\n"
      ".inst 0x6e44ec7a  // bfmmla v26.4s, v3.8h, v4.8h\n"
      ".inst 0x6e45ec7d  // bfmmla v29.4s, v3.8h, v5.8h\n"
      "ldr q4, [x20, #0x20]\n"
      "ldr q5, [x20, #0x30]\n"
      ".inst 0x6e46ec09  // bfmmla v9.4s, v0.8h, v6.8h\n"
      ".inst 0x6e46ec2f  // bfmmla v15.4s, v1.8h, v6.8h\n"
      ".inst 0x6e46ec55  // bfmmla v21.4s, v2.8h, v6.8h\n"
      ".inst 0x6e46ec7b  // bfmmla v27.4s, v3.8h, v6.8h\n"
      "add %x[Apanel], %x[Apanel], #0x10\n"
      ".inst 0x6e47ec0c  // bfmmla v12.4s, v0.8h, v7.8h\n"
      ".inst 0x6e44ec0a  // bfmmla v10.4s, v0.8h, v4.8h\n"
      "add x20, x20, #0x40\n"
      ".inst 0x6e45ec0d  // bfmmla v13.4s, v0.8h, v5.8h\n"
      ".inst 0x6e47ec32  // bfmmla v18.4s, v1.8h, v7.8h\n"
      ".inst 0x6e44ec30  // bfmmla v16.4s, v1.8h, v4.8h\n"
      ".inst 0x6e45ec33  // bfmmla v19.4s, v1.8h, v5.8h\n"
      ".inst 0x6e47ec58  // bfmmla v24.4s, v2.8h, v7.8h\n"
      ".inst 0x6e47ec7e  // bfmmla v30.4s, v3.8h, v7.8h\n"
      ".inst 0x6e44ec56  // bfmmla v22.4s, v2.8h, v4.8h\n"
      ".inst 0x6e45ec59  // bfmmla v25.4s, v2.8h, v5.8h\n"
      ".inst 0x6e44ec7c  // bfmmla v28.4s, v3.8h, v4.8h\n"
      ".inst 0x6e45ec7f  // bfmmla v31.4s, v3.8h, v5.8h\n"
      "cbz x19, 5f\n"
      "ldr q6, [x20, #0x0]\n"
      "ldr q0, [%x[Apanel], #0x0]\n"
      ".inst 0x6e46ec08  // bfmmla v8.4s, v0.8h, v6.8h\n"
      "ldr q1, [%x[Apanel], #0x10]\n"
      "ldr q7, [x20, #0x10]\n"
      ".inst 0x6e46ec2e  // bfmmla v14.4s, v1.8h, v6.8h\n"
      "ldr q2, [%x[Apanel], #0x20]\n"
      "ldr q3, [%x[Apanel], #0x30]\n"
      ".inst 0x6e47ec0b  // bfmmla v11.4s, v0.8h, v7.8h\n"
      ".inst 0x6e47ec31  // bfmmla v17.4s, v1.8h, v7.8h\n"
      ".inst 0x6e46ec54  // bfmmla v20.4s, v2.8h, v6.8h\n"
      "ldr q4, [x20, #0x20]\n"
      ".inst 0x6e47ec57  // bfmmla v23.4s, v2.8h, v7.8h\n"
      ".inst 0x6e46ec7a  // bfmmla v26.4s, v3.8h, v6.8h\n"
      "ldr q5, [x20, #0x30]\n"
      ".inst 0x6e47ec7d  // bfmmla v29.4s, v3.8h, v7.8h\n"
      "ldr q6, [x20, #0x40]\n"
      "ldr q7, [x20, #0x50]\n"
      ".inst 0x6e44ec09  // bfmmla v9.4s, v0.8h, v4.8h\n"
      ".inst 0x6e44ec2f  // bfmmla v15.4s, v1.8h, v4.8h\n"
      "add x20, x20, #0x60\n"
      ".inst 0x6e44ec55  // bfmmla v21.4s, v2.8h, v4.8h\n"
      ".inst 0x6e44ec7b  // bfmmla v27.4s, v3.8h, v4.8h\n"
      "add %x[Apanel], %x[Apanel], #0x40\n"
      ".inst 0x6e45ec0c  // bfmmla v12.4s, v0.8h, v5.8h\n"
      ".inst 0x6e46ec0a  // bfmmla v10.4s, v0.8h, v6.8h\n"
      ".inst 0x6e47ec0d  // bfmmla v13.4s, v0.8h, v7.8h\n"
      ".inst 0x6e45ec32  // bfmmla v18.4s, v1.8h, v5.8h\n"
      ".inst 0x6e46ec30  // bfmmla v16.4s, v1.8h, v6.8h\n"
      ".inst 0x6e47ec33  // bfmmla v19.4s, v1.8h, v7.8h\n"
      ".inst 0x6e45ec58  // bfmmla v24.4s, v2.8h, v5.8h\n"
      ".inst 0x6e45ec7e  // bfmmla v30.4s, v3.8h, v5.8h\n"
      ".inst 0x6e46ec56  // bfmmla v22.4s, v2.8h, v6.8h\n"
      ".inst 0x6e47ec59  // bfmmla v25.4s, v2.8h, v7.8h\n"
      ".inst 0x6e46ec7c  // bfmmla v28.4s, v3.8h, v6.8h\n"
      ".inst 0x6e47ec7f  // bfmmla v31.4s, v3.8h, v7.8h\n"
      "5:"  // multiply loop done
      "subs x22, x22, #0x1\n"
      "uzp1 v4.2d, v8.2d, v11.2d\n"
      "uzp2 v8.2d, v8.2d, v11.2d\n"
      "uzp1 v11.2d, v9.2d, v12.2d\n"
      "uzp2 v9.2d, v9.2d, v12.2d\n"
      "str q4, [%x[Cpanel], #0x0]\n"
      "uzp1 v12.2d, v10.2d, v13.2d\n"
      "uzp2 v10.2d, v10.2d, v13.2d\n"
      "str q11, [%x[Cpanel], #0x10]\n"
      "str q12, [%x[Cpanel], #0x20]\n"
      "uzp1 v13.2d, v14.2d, v17.2d\n"
      "uzp2 v14.2d, v14.2d, v17.2d\n"
      "str q8, [%x[Cpanel], #0x30]\n"
      "uzp1 v17.2d, v15.2d, v18.2d\n"
      "uzp2 v15.2d, v15.2d, v18.2d\n"
      "str q9, [%x[Cpanel], #0x40]\n"
      "uzp1 v18.2d, v16.2d, v19.2d\n"
      "uzp2 v16.2d, v16.2d, v19.2d\n"
      "str q10, [%x[Cpanel], #0x50]\n"
      "uzp1 v19.2d, v20.2d, v23.2d\n"
      "uzp2 v20.2d, v20.2d, v23.2d\n"
      "str q13, [%x[Cpanel], #0x60]\n"
      "uzp1 v23.2d, v21.2d, v24.2d\n"
      "uzp2 v21.2d, v21.2d, v24.2d\n"
      "str q17, [%x[Cpanel], #0x70]\n"
      "uzp1 v24.2d, v22.2d, v25.2d\n"
      "uzp2 v22.2d, v22.2d, v25.2d\n"
      "str q18, [%x[Cpanel], #0x80]\n"
      "uzp1 v25.2d, v26.2d, v29.2d\n"
      "uzp2 v26.2d, v26.2d, v29.2d\n"
      "str q14, [%x[Cpanel], #0x90]\n"
      "uzp1 v29.2d, v27.2d, v30.2d\n"
      "uzp2 v27.2d, v27.2d, v30.2d\n"
      "str q15, [%x[Cpanel], #0xa0]\n"
      "uzp1 v30.2d, v28.2d, v31.2d\n"
      "uzp2 v28.2d, v28.2d, v31.2d\n"
      "str q16, [%x[Cpanel], #0xb0]\n"
      "str q19, [%x[Cpanel], #0xc0]\n"
      "str q23, [%x[Cpanel], #0xd0]\n"
      "str q24, [%x[Cpanel], #0xe0]\n"
      "str q20, [%x[Cpanel], #0xf0]\n"
      "str q21, [%x[Cpanel], #0x100]\n"
      "str q22, [%x[Cpanel], #0x110]\n"
      "str q25, [%x[Cpanel], #0x120]\n"
      "str q29, [%x[Cpanel], #0x130]\n"
      "str q30, [%x[Cpanel], #0x140]\n"
      "str q26, [%x[Cpanel], #0x150]\n"
      "str q27, [%x[Cpanel], #0x160]\n"
      "str q28, [%x[Cpanel], #0x170]\n"
      "add %x[Cpanel], %x[Cpanel], #0x180\n"
      "bgt 2b\n"
      "subs %x[ablocks], %x[ablocks], #0x1\n"
      "bne 1b\n"
      : [Apanel] "+&r" (Apanel), [Cpanel] "+&r" (Cpanel), [ablocks] "+&r" (ablocks)
      : [args_ptr] "r" (&ka), [offsetof_Bpanel] "I" (offsetof(KernelArgs, Bpanel)), [offsetof_K] "I" (offsetof(KernelArgs, K)), [offsetof_bblocks] "I" (offsetof(KernelArgs, bblocks))
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x19", "x20", "x21", "x22"
    );
}

void merge_12x8(float *C, int ldc, float *sc) {
    float *outptr0 = C;
    float *outptr1 = outptr0 + ldc;
    float *outptr2 = outptr1 + ldc;
    float *outptr3 = outptr2 + ldc;
    float *outptr4 = outptr3 + ldc;
    float *outptr5 = outptr4 + ldc;
    float *outptr6 = outptr5 + ldc;
    float *outptr7 = outptr6 + ldc;
    float minval = - static_cast<float>(std::numeric_limits<float>::infinity());
    float maxval =   static_cast<float>(std::numeric_limits<float>::infinity());

    /* Optimized routine to copy an entire block */
    __asm __volatile(
        "dup v0.4s, %[maxval].s[0]\n"
        "ldr q2, [%[outptr0]]\n"
        "dup v1.4s, %[minval].s[0]\n"
        "ldr q10, [%[inptr]]\n"
        "ldr q3, [%[outptr0], #0x10]\n"
        "prfm PLDL1KEEP, [%[inptr], #0x180]\n"
        "ldr q11, [%[inptr], #0x10]\n"
        "prfm PLDL1KEEP, [%[outptr0], #0x60]\n"
        "fadd v10.4s, v10.4s, v2.4s\n"
        "ldr q4, [%[outptr0], #0x20]\n"
        "ldr q12, [%[inptr], #0x20]\n"
        "prfm PLDL1KEEP, [%[inptr], #0x1c0]\n"
        "fadd v11.4s, v11.4s, v3.4s\n"
        "ldr q5, [%[outptr1]]\n"
        "fmin v10.4s, v10.4s, v0.4s\n"
        "ldr q13, [%[inptr], #0x30]\n"
        "fadd v12.4s, v12.4s, v4.4s\n"
        "ldr q6, [%[outptr1], #0x10]\n"
        "ldr q14, [%[inptr], #0x40]\n"
        "prfm PLDL1KEEP, [%[outptr1], #0x60]\n"
        "fmax v10.4s, v10.4s, v1.4s\n"
        "ldr q7, [%[outptr1], #0x20]\n"
        "fmin v11.4s, v11.4s, v0.4s\n"
        "ldr q15, [%[inptr], #0x50]\n"
        "fmin v12.4s, v12.4s, v0.4s\n"
        "ldr q8, [%[outptr2]]\n"
        "fadd v13.4s, v13.4s, v5.4s\n"
        "str q10, [%[outptr0]]\n"
        "fadd v14.4s, v14.4s, v6.4s\n"
        "ldr q16, [%[inptr], #0x60]\n"
        "fmax v11.4s, v11.4s, v1.4s\n"
        "ldr q9, [%[outptr2], #0x10]\n"
        "fmax v12.4s, v12.4s, v1.4s\n"
        "ldr q17, [%[inptr], #0x70]\n"
        "fmin v13.4s, v13.4s, v0.4s\n"
        "ldr q2, [%[outptr2], #0x20]\n"
        "fmin v14.4s, v14.4s, v0.4s\n"
        "str q11, [%[outptr0], #0x10]\n"
        "fadd v15.4s, v15.4s, v7.4s\n"
        "ldr q10, [%[inptr], #0x80]\n"
        "fadd v16.4s, v16.4s, v8.4s\n"
        "ldr q3, [%[outptr3]]\n"
        "fmax v13.4s, v13.4s, v1.4s\n"
        "str q12, [%[outptr0], #0x20]\n"
        "fmax v14.4s, v14.4s, v1.4s\n"
        "ldr q11, [%[inptr], #0x90]\n"
        "fmin v15.4s, v15.4s, v0.4s\n"
        "ldr q4, [%[outptr3], #0x10]\n"
        "fmin v16.4s, v16.4s, v0.4s\n"
        "str q13, [%[outptr1]]\n"
        "fadd v17.4s, v17.4s, v9.4s\n"
        "ldr q12, [%[inptr], #0xa0]\n"
        "fadd v10.4s, v10.4s, v2.4s\n"
        "ldr q5, [%[outptr3], #0x20]\n"
        "fmax v15.4s, v15.4s, v1.4s\n"
        "str q14, [%[outptr1], #0x10]\n"
        "fmax v16.4s, v16.4s, v1.4s\n"
        "ldr q13, [%[inptr], #0xb0]\n"
        "fmin v17.4s, v17.4s, v0.4s\n"
        "ldr q6, [%[outptr4]]\n"
        "fmin v10.4s, v10.4s, v0.4s\n"
        "str q15, [%[outptr1], #0x20]\n"
        "fadd v11.4s, v11.4s, v3.4s\n"
        "ldr q14, [%[inptr], #0xc0]\n"
        "fadd v12.4s, v12.4s, v4.4s\n"
        "ldr q7, [%[outptr4], #0x10]\n"
        "fmax v17.4s, v17.4s, v1.4s\n"
        "str q16, [%[outptr2]]\n"
        "fmax v10.4s, v10.4s, v1.4s\n"
        "ldr q15, [%[inptr], #0xd0]\n"
        "fmin v11.4s, v11.4s, v0.4s\n"
        "ldr q8, [%[outptr4], #0x20]\n"
        "fmin v12.4s, v12.4s, v0.4s\n"
        "str q17, [%[outptr2], #0x10]\n"
        "fadd v13.4s, v13.4s, v5.4s\n"
        "ldr q16, [%[inptr], #0xe0]\n"
        "fadd v14.4s, v14.4s, v6.4s\n"
        "ldr q9, [%[outptr5]]\n"
        "fmax v11.4s, v11.4s, v1.4s\n"
        "str q10, [%[outptr2], #0x20]\n"
        "fmax v12.4s, v12.4s, v1.4s\n"
        "ldr q17, [%[inptr], #0xf0]\n"
        "fmin v13.4s, v13.4s, v0.4s\n"
        "ldr q2, [%[outptr5], #0x10]\n"
        "fmin v14.4s, v14.4s, v0.4s\n"
        "str q11, [%[outptr3]]\n"
        "fadd v15.4s, v15.4s, v7.4s\n"
        "ldr q10, [%[inptr], #0x100]\n"
        "fadd v16.4s, v16.4s, v8.4s\n"
        "ldr q3, [%[outptr5], #0x20]\n"
        "fmax v13.4s, v13.4s, v1.4s\n"
        "str q12, [%[outptr3], #0x10]\n"
        "fmax v14.4s, v14.4s, v1.4s\n"
        "ldr q11, [%[inptr], #0x110]\n"
        "fmin v15.4s, v15.4s, v0.4s\n"
        "ldr q4, [%[outptr6]]\n"
        "fmin v16.4s, v16.4s, v0.4s\n"
        "str q13, [%[outptr3], #0x20]\n"
        "fadd v17.4s, v17.4s, v9.4s\n"
        "ldr q12, [%[inptr], #0x120]\n"
        "fadd v10.4s, v10.4s, v2.4s\n"
        "ldr q5, [%[outptr6], #0x10]\n"
        "fmax v15.4s, v15.4s, v1.4s\n"
        "str q14, [%[outptr4]]\n"
        "fmax v16.4s, v16.4s, v1.4s\n"
        "ldr q13, [%[inptr], #0x130]\n"
        "fmin v17.4s, v17.4s, v0.4s\n"
        "ldr q6, [%[outptr6], #0x20]\n"
        "fmin v10.4s, v10.4s, v0.4s\n"
        "str q15, [%[outptr4], #0x10]\n"
        "fadd v11.4s, v11.4s, v3.4s\n"
        "ldr q14, [%[inptr], #0x140]\n"
        "fadd v12.4s, v12.4s, v4.4s\n"
        "ldr q7, [%[outptr7]]\n"
        "fmax v17.4s, v17.4s, v1.4s\n"
        "str q16, [%[outptr4], #0x20]\n"
        "fmax v10.4s, v10.4s, v1.4s\n"
        "ldr q15, [%[inptr], #0x150]\n"
        "fmin v11.4s, v11.4s, v0.4s\n"
        "ldr q8, [%[outptr7], #0x10]\n"
        "fmin v12.4s, v12.4s, v0.4s\n"
        "str q17, [%[outptr5]]\n"
        "fadd v13.4s, v13.4s, v5.4s\n"
        "ldr q16, [%[inptr], #0x160]\n"
        "fadd v14.4s, v14.4s, v6.4s\n"
        "ldr q9, [%[outptr7], #0x20]\n"
        "fmax v11.4s, v11.4s, v1.4s\n"
        "str q10, [%[outptr5], #0x10]\n"
        "fmax v12.4s, v12.4s, v1.4s\n"
        "ldr q17, [%[inptr], #0x170]\n"
        "fmin v13.4s, v13.4s, v0.4s\n"
        "add %[outptr0], %[outptr0], #0x30\n"
        "fmin v14.4s, v14.4s, v0.4s\n"
        "str q11, [%[outptr5], #0x20]\n"
        "fadd v15.4s, v15.4s, v7.4s\n"
        "add %[outptr1], %[outptr1], #0x30\n"
        "fmax v13.4s, v13.4s, v1.4s\n"
        "str q12, [%[outptr6]]\n"
        "fmax v14.4s, v14.4s, v1.4s\n"
        "prfm PLDL1KEEP, [%[inptr], #0x200]\n"
        "fmin v15.4s, v15.4s, v0.4s\n"
        "str q13, [%[outptr6], #0x10]\n"
        "fadd v16.4s, v16.4s, v8.4s\n"
        "prfm PLDL1KEEP, [%[outptr2], #0x60]\n"
        "fadd v17.4s, v17.4s, v9.4s\n"
        "str q14, [%[outptr6], #0x20]\n"
        "fmax v15.4s, v15.4s, v1.4s\n"
        "add %[outptr2], %[outptr2], #0x30\n"
        "fmin v16.4s, v16.4s, v0.4s\n"
        "prfm PLDL1KEEP, [%[outptr3], #0x60]\n"
        "fmin v17.4s, v17.4s, v0.4s\n"
        "str q15, [%[outptr7]]\n"
        "add %[outptr3], %[outptr3], #0x30\n"
        "fmax v16.4s, v16.4s, v1.4s\n"
        "prfm PLDL1KEEP, [%[inptr], #0x240]\n"
        "fmax v17.4s, v17.4s, v1.4s\n"
        "prfm PLDL1KEEP, [%[outptr4], #0x60]\n"
        "str q16, [%[outptr7], #0x10]\n"
        "add %[outptr4], %[outptr4], #0x30\n"
        "prfm PLDL1KEEP, [%[inptr], #0x280]\n"
        "prfm PLDL1KEEP, [%[outptr5], #0x60]\n"
        "str q17, [%[outptr7], #0x20]\n"
        "add %[outptr5], %[outptr5], #0x30\n"
        "prfm PLDL1KEEP, [%[inptr], #0x2c0]\n"
        "prfm PLDL1KEEP, [%[outptr6], #0x60]\n"
        "add %[outptr6], %[outptr6], #0x30\n"
        "prfm PLDL1KEEP, [%[outptr7], #0x60]\n"
        "add %[outptr7], %[outptr7], #0x30\n"
        "add %[inptr], %[inptr], #0x180\n"
        : [outptr0] "+r"(outptr0), [outptr1] "+r"(outptr1), [outptr2] "+r"(outptr2),
          [outptr3] "+r"(outptr3), [outptr4] "+r"(outptr4), [outptr5] "+r"(outptr5),
          [outptr6] "+r"(outptr6), [outptr7] "+r"(outptr7), [inptr] "+r"(sc)
        : [minval] "w"(minval), [maxval] "w"(maxval)
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13",
          "v14", "v15", "v16", "v17", "v18", "v19", "v20", "memory");
}

void kernel(int M, int K, int N, bfloat16 *sa, bfloat16 *sb, float *C, int ldc) {
    float *tmpc = (float *)malloc(12 * 8 * sizeof(float));
    for (int i = 0; i < M; i += 8) {
        for (int j = 0; j < N; j += 12) {
            a64_interleaved_bf16fp32_mmla_8x12(sa, sb, tmpc, 1, 1, K);
            merge_12x8(C + i * ldc + j, ldc, tmpc);
        }
    }
    free(tmpc);
}

int my_impl(int M, int K, int N, bfloat16 *A, bfloat16 *B, float *C) {
#ifdef DEBUG
    constexpr int BLOCK_M = 8;
    constexpr int BLOCK_K = 12;
    constexpr int BLOCK_N = 12;
#else
    constexpr int BLOCK_M = 256;
    constexpr int BLOCK_K = 256;
    constexpr int BLOCK_N = 240;
#endif

    bfloat16 *buffer = (bfloat16 *)malloc((BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * sizeof(bfloat16));

    bfloat16 *sa = buffer;
    bfloat16 *sb = buffer + BLOCK_M * BLOCK_K;

    for (int i = 0; i < M; i += BLOCK_M) {
        for (int p = 0; p < K; p += BLOCK_K) {
            packA(BLOCK_M, BLOCK_K, A + i * K + p, K, sa);
            for (int j = 0; j < N; j += BLOCK_N) {
                packB(BLOCK_K, BLOCK_N, B + p * N + j, N, sb);
                kernel(BLOCK_M, BLOCK_K, BLOCK_N, sa, sb, C + i * N + j, N);
                // kernel(BLOCK_M, BLOCK_K, BLOCK_N, sa, sb, tmpc);
                // merge(BLOCK_M, BLOCK_N, tmpc, C + j * M + i, M);
            }
        }
    }

    free(buffer);
    return 0;
}

int check() {
#ifdef DEBUG
    constexpr int SIZE = 12;
#else
    constexpr int SIZE = 2400;
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
    display_matrix(FA, M, K, ArrangeMode::RowMajor);
    printf("Matrix B:\n");
    display_matrix(FB, K, N, ArrangeMode::RowMajor);
#endif

    native_c(M, K, N, A, B, refc);
    my_impl(M, K, N, A, B, myc);
    compare_array(refc, myc, M * N);

#ifdef DEBUG
    printf("Matrix refc:\n");
    display_matrix(refc, M, N, ArrangeMode::RowMajor);
    printf("Matrix myc:\n");
    display_matrix(myc, M, N, ArrangeMode::RowMajor);
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
 * row-major matrix
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
