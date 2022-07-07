#include <immintrin.h>

#define BLOCK 8
#define BLOCK_M 4
#define BLOCK_N 2


int my_impl(int M, int K, int N, float *A, int lda, float *B, int ldb, float *C, int ldc) {
    
    
    for (int i = 0; i < M; i += BLOCK_M) {
        for (int j = 0; j < N; j += BLOCK * BLOCK_N) {
            __m256 acc[BLOCK_M][BLOCK_N] = {};
            for (int k = 0; k < K; k++) {
                for (int im = 0; im < BLOCK_M; im++) {
                    __m256 ta = _mm256_broadcast_ss(&A[(i + im) * lda + k]);
                    for (int in = 0; in < BLOCK_N; in++) {
                        acc[im][in] = 
                            _mm256_fmadd_ps(ta, , acc[im][in]);
                    }
                }
            }
        }
    }

    // float *sa = new float[BLOCK_M * BLOCK_K]; 
    // float *sb = new float[BLOCK_K * BLOCK_N];

    // for (int bm = 0; bm < M; bm += BLOCK_M) {
    //     for (int bk = 0; bk < K; bk += BLOCK_K) {
    //         packA_4(BLOCK_M, BLOCK_K, A + bk * M + bm, lda, sa);
    //         for (int bn = 0; bn < N; bn += BLOCK_N) {
    //             packB_4(BLOCK_K, BLOCK_N, B + bn * K + bk, ldb, sb);
    //             kernel(BLOCK_M, BLOCK_K, BLOCK_N, sa, sb, C + bn * M + bm, M);
    //         }
    //     }
    // }

    // delete[] sa;
    // delete[] sb;
    return 0;
}
