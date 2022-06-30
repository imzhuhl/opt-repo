// 1.160 GFLOP/S, 925.63 ms
int my_impl(int M, int K, int N, float *A, int lda, float *B, int ldb, float *C, int ldc) {
    constexpr int block_size = 16;
    for (int bm = 0; bm < M; bm += block_size) {
        for (int km = 0; km < K; km += block_size) {
            
            for (int i = bm; i < bm + block_size; i++) {
                for (int j = 0; j < N; j++) {
                    float tmp = 0.0;
                    for (int p = km; p < km + block_size; p++) {
                        tmp += A[lda * p + i] * B[ldb * j + p];
                    }
                    C[ldc * j + i] += tmp;
                }
            }

        }
    }
    return 0;
}
