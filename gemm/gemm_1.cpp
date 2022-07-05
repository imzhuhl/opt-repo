
int my_impl(int M, int K, int N, float *A, int lda, float *B, int ldb, float *C, int ldc) {
    constexpr int block_size = 4;
    for (int bm = 0; bm < M; bm += block_size) {
        for (int bk = 0; bk < K; bk += block_size) {
            for (int bn = 0; bn < N; bn += block_size) {

                for (int i = bm; i < bm + block_size; i++) {
                    for (int j = bn; j < bn + block_size; j++) {
                        float tmp = 0.0;
                        for (int p = bk; p < bk + block_size; p++) {
                            tmp += A[lda * p + i] * B[ldb * j + p];
                        }
                        C[ldc * j + i] += tmp;
                    }
                }

            }
        }
    }
    return 0;
}
