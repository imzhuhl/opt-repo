#pragma once

#include <random>
#include <cmath>

using bfloat16 = uint16_t;

enum class InitVecFlag {
    Zero,
    One,
    IncreaseByOne,
    RandonValue,
};

enum class ArrangeMode {
    ColMajor,
    RowMajor,
};

void fill_array(float *v, int length, InitVecFlag flag) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(-1.0, 1.0);

    for (size_t i = 0; i < length; i++) {
        switch (flag)
        {
        case InitVecFlag::Zero : 
            v[i] = 0;
            break;
        case InitVecFlag::One : 
            v[i] = 1;
            break;
        case InitVecFlag::IncreaseByOne : 
            v[i] = i;
            break;
        case InitVecFlag::RandonValue :
            v[i] = dist(mt);
            break;
        default:
            printf("Error InitVecFlag value.\n");
            exit(1);
        }
    }
}

/**
 * Display a matrix
 * @param M     number of rows of matrix
 * @param N     number of columns of matrix
 * @param flag  elements arrangement
 */
void display_matrix(float *mat, int M, int N, ArrangeMode flag) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (flag == ArrangeMode::RowMajor) {
                printf(" %7.2f", mat[i * N + j]);
            } else if (flag == ArrangeMode::ColMajor) {
                printf(" %7.2f", mat[j * M + i]);
            }
        }
        printf("\n");
    }
    printf("\n");
}

void compare_array(float *a, float *b, int size) {
    float diff = 0.0;
    for (int i = 0; i < size; i++) {
        diff = std::abs(a[i] - b[i]);
        if (diff > 1e-3) {
            printf("Check error: %.2f vs %.2f\n", a[i], b[i]);
            return;
        }
    }
    printf("Check pass.\n");
    return;
}


void copy_array(float *src, float *dst, int size) {
    for (int i = 0; i < size; i++) {
        dst[i] = src[i];
    }
}


void array_fp32_to_bf16(float *src, bfloat16 *dst, size_t size) {
    for (size_t i = 0; i < size; i++) {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        dst[i] = *(reinterpret_cast<bfloat16 *>(&src[i]));
#else
        dst[i] = *(reinterpret_cast<bfloat16 *>(&src[i])+1);
#endif
    }
}
