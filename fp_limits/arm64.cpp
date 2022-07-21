#include <arm_sve.h>
#include <chrono>
#include <iostream>

void fmla_kernal(int cnt) {
    __asm__ __volatile__(
        "mov x0, %[cnt]\n"
        "ptrue p0.b\n"
        "1:\n"
        "fmla z10.s, p0/M, z0.s, z2.s\n"
        "fmla z11.s, p0/M, z0.s, z3.s\n"
        "fmla z12.s, p0/M, z0.s, z4.s\n"
        "fmla z13.s, p0/M, z0.s, z5.s\n"
        "fmla z14.s, p0/M, z1.s, z2.s\n"
        "fmla z15.s, p0/M, z1.s, z3.s\n"
        "fmla z16.s, p0/M, z1.s, z4.s\n"
        "fmla z17.s, p0/M, z1.s, z5.s\n"
        "subs x0, x0, #1\n"
        "bne 1b\n"
        :
        : [cnt] "r"(cnt)
        : "cc", "memory", "x0", "p0", "z0", "z1", "z2", "z3", "z4", "z5", "z10", "z11", "z12",
          "z13", "z14", "z15", "z16", "z17");
}

/** Peak performance of the FLMA instruction
 *
 * fmla dst, a, b
 *  -> dst = dst + a * b
 *
 * Length of vector register: 128 bit
 * Data type: float, 32 bit
 *
 * One FMLA inst: 8 FLOPs (4 * 2)
 *
 */
void run_fmla() {
    int cnt = 2048 * 2048;
    int fmla_inst_num = 8;
    int flop_per_fmla = 8;
    double flop = cnt * fmla_inst_num * flop_per_fmla;

    double best_gflops = 0.0;
    for (int rep = 0; rep < 5; rep++) {
        auto st = std::chrono::steady_clock::now();
        fmla_kernal(cnt);
        auto et = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> elapsed = et - st;

        double gflops = (flop * 1e-9) / (elapsed.count() * 1e-3);
        if (best_gflops < gflops) {
            best_gflops = gflops;
        }
    }
    printf("FMLA: %.3lf GFLOP/S\n", best_gflops);
}

void bfmmla_kernel(int cnt) {
    __asm__ __volatile__(
        "mov x0, %[cnt]\n"
        "ptrue p0.b\n"
        "1:\n"
        "bfmmla z10.s, z5.h, z0.h\n"
        "bfmmla z11.s, z5.h, z1.h\n"
        "bfmmla z12.s, z5.h, z2.h\n"
        "bfmmla z13.s, z5.h, z3.h\n"
        "bfmmla z14.s, z5.h, z4.h\n"
        "bfmmla z15.s, z6.h, z0.h\n"
        "bfmmla z16.s, z6.h, z1.h\n"
        "bfmmla z17.s, z6.h, z2.h\n"
        "bfmmla z18.s, z6.h, z3.h\n"
        "bfmmla z19.s, z6.h, z4.h\n"
        "subs x0, x0, #1\n"
        "bne 1b\n"
        :
        : [cnt] "r"(cnt)
        : "cc", "memory", "x0", "p0", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z10", "z11", "z12",
          "z13", "z14", "z15", "z16", "z17", "z18", "z19");
}

/** Peak performance of the FLMA instruction
 *
 * Length of vector register: 128 bit
 * Data type: bfloat16, 16 bit
 *
 * One BFMMLA inst: 32 FLOPs
 *
 */
void run_bfmmla() {
    int cnt = 2048 * 2048;
    int bfmmla_inst_num = 10;
    int flop_per_bfmmla = 32;
    double flop = cnt * bfmmla_inst_num * flop_per_bfmmla;

    double best_gflops = 0.0;
    for (int rep = 0; rep < 5; rep++) {
        auto st = std::chrono::steady_clock::now();
        bfmmla_kernel(cnt);
        auto et = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> elapsed = et - st;

        double gflops = (flop * 1e-9) / (elapsed.count() * 1e-3);
        if (best_gflops < gflops) {
            best_gflops = gflops;
        }
    }
    printf("BFMMLA: %.3lf GFLOP/S\n", best_gflops);
}

int main() {
    run_fmla();
    run_bfmmla();
    return 0;
}
