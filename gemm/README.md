Matrix Multiplication

My goal is to compare numpy's implementation, which uses the OpenBLAS backend.

* gemm_1: matrix block
* gemm_2: pack matrix
* gemm_3: use simd instructions (NEON)
* gemm_4: use 4x4 kernel, which means use 1 + 4 vector registors

I count FLOPS and execution time, and the reults are recorded in file `result.txt`
