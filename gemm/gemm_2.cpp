#include <iostream>

void packA(int M, int N, float *A, int lda, float *sa) {
    float *ptr_sa = sa;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            *ptr_sa = A[j * lda + i];            
            ptr_sa++;
        }
    }

}

void packB(int K, int N, float *B, int ldb, float *sb) {
    float *ptr_sb = sb;
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < K; i++) {
            *ptr_sb = B[j * ldb + i];
            ptr_sb++;
        }
    }
}

void kernel(int M, int K, int N, float *A, float *B, float *C, int ldc) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float tmp = 0;
            for (int p = 0; p < K; p++) {
                tmp += A[i * K + p] * B[j * K + p];
            }
            C[j * ldc + i] += tmp;
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
